from fastapi import FastAPI, Request, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from fastapi.responses import StreamingResponse
from io import BytesIO
from io import StringIO
import argparse
from database.db import Database
from database.models import Model
import multiprocessing
import base64
from pydantic import BaseModel
import asyncio
import os
import torch
from torchvision import transforms
from transformers import DeiTFeatureExtractor
import zipfile
from database import dataset
import faiss
from cytomine import Cytomine
from cytomine.models.image import ImageInstance
from cytomine.models import AnnotationCollection
from cytomine.models import TermCollection
from cytomine.models import CurrentUser
from openslide import OpenSlide
import cv2
import numpy as np
from database.dataset import AddSlide
from sklearn.cluster import KMeans
import requests
from readerwriterlock import rwlock_async
import concurrent.futures

app = FastAPI()

class Server:
    def __init__(self, ip, port, master_ip, master_port, model, num_features,
                 weights, use_dr, device, image_folder, http, db_name,
                 host, name):
        self.ip = ip
        self.port = port
        self.master_ip = master_ip
        self.master_port = master_port

        self.model = Model(model, num_features=num_features, name=weights,
                           use_dr=use_dr, device=device)

        self.database = Database(db_name, self.model, load=True,
                                 transformer = model=='transformer', device='cpu')
        self.image_folder = image_folder
        self.host = host
        self.name = name

        self.pool = concurrent.futures.ThreadPoolExecutor()

        requests.get('{}://{}:{}/connect'.format('http' if http else 'https', master_ip, master_port),
                     params={'ip': '{}:{}'.format(self.ip, self.port), 'nbr_images_labeled': self.database.index_labeled.ntotal,
                             'nbr_images_unlabeled': self.database.index_unlabeled.ntotal})

parser = argparse.ArgumentParser()

parser.add_argument(
    '--master_ip',
    default='127.0.0.1'
)

parser.add_argument(
    '--master_port',
    default=8000,
    type=int
)

parser.add_argument(
    '--ip',
    default='127.0.0.1'
)

parser.add_argument(
    '--port',
    default=8001,
    type=int
)

parser.add_argument(
    '--extractor',
    default='densenet'
)

parser.add_argument(
    '--num_features',
    default=128,
    type=int
)

parser.add_argument(
    '--weights',
    default='weights/weights'
)

parser.add_argument(
    '--use_dr',
    action='store_true'
)

parser.add_argument(
    '--gpu_id',
    default=0,
    type=int
)

parser.add_argument(
    '--folder',
    default='images'
)

parser.add_argument(
    '--http',
    action='store_true'
)

parser.add_argument(
    '--db_name',
    default='db'
)

parser.add_argument(
    '--host'
)

parser.add_argument(
    '--server_name'
)

args = parser.parse_args()

if args.gpu_id >= 0:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

if __name__ != '__main__':
    server = Server(args.ip, args.port, args.master_ip, args.master_port,
                    args.extractor, args.num_features, args.weights, args.use_dr,
                    device, args.folder, args.http, args.db_name,
                    args.host, args.server_name)

    names = {}

    lock = rwlock_async.RWLockRead()

@app.post('/nearest_neighbours/{retrieve_class}')
async def nearest_neighbours(nrt_neigh: int, public_key: str, private_key: str, retrieve_class: str, image: UploadFile=File(...)):
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB')
    loop = asyncio.get_running_loop()

    async with await lock.gen_rlock():
        names[public_key], distance, _, _ = await loop.run_in_executor(
            server.pool, server.database.search, img, nrt_neigh, retrieve_class)

    return {'distances': distance[0]}

class LabelList(BaseModel):
    labels: list

@app.get('/retrieve_images/{retrieve_class}')
def retrieve_images(retrieve_class: str, labels: LabelList, public_key: str, private_key: str):
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    images = []
    img_names = []
    if retrieve_class == 'true':
        cls = []

        for i in labels.labels:
            img = Image.open(names[public_key][int(i)])
            img_names.append(names[public_key][int(i)])
            end = names[public_key][int(i)].rfind("/")
            begin = names[public_key][int(i)].rfind("/", 0, end) + 1
            cls.append(names[public_key][int(i)][begin: end])
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))

        return {'images': images, 'cls': cls, 'names': img_names}
    elif retrieve_class == 'false':
        for i in labels.labels:
            print(i, names[public_key])
            img = Image.open(names[public_key][int(i)])
            img_names.append(names[public_key][int(i)])
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))
        return {'images': images, 'cls': ["Unkown" for i in images], 'names': img_names}
    elif retrieve_class == 'mix':
        cls = []

        for i in labels.labels:
            img = Image.open(names[public_key][int(i)])
            img_names.append(names[public_key][int(i)])
            if names[public_key][int(i)].count('/') > 1:
                end = names[public_key][int(i)].rfind("/")
                begin = names[public_key][int(i)].rfind("/", 0, end) + 1
                cls.append(names[public_key][int(i)][begin: end])
            else:
                cls.append('Unkown')
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))

        return {'images': images, 'cls': cls, 'names': img_names}

@app.post('/index_image')
async def index_image(public_key: str, private_key: str, image: UploadFile=File(...), label: str=''):
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB')
    name = image.filename[image.filename.rfind('/')+1: image.filename.rfind('.')] + '.png'

    async with await lock.gen_wlock():
        last_id = server.database.r.get('last_id_{}'.format('labeled' if label != '' else 'unlabeled')).decode('utf-8')
        print(last_id)
        if not os.path.exists(os.path.join(server.image_folder, label)):
            os.makedirs(os.path.join(server.image_folder, label))
        img.save(os.path.join(server.image_folder, label, server.name + '_' + last_id + '.png'), 'PNG')

        with torch.no_grad():
            if not server.database.feat_extract:
                image = transforms.Resize((224, 224))(img)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(image)
            else:
                image = server.database.feat_extract(images=img, return_tensors='pt')['pixel_values']

            out = server.model(image.to(device=next(server.model.parameters()).device).view(1, 3, 224, 224))

            server.database.add(out.cpu().numpy().reshape(-1, server.model.num_features),
                                [os.path.join(server.image_folder, label, server.name + '_' + last_id + '.png')], label!='')

            server.database.save()
    return

@app.post('/index_folder')
async def index_folder(labeled: bool, public_key: str, private_key: str, folder: UploadFile=File(...)):
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    content = await folder.read()

    try:
        zf = zipfile.ZipFile(BytesIO(content))
    except zipfile.BadZipFile as e:
        raise HTTPException(401, str(e))
    name_list = zf.namelist()

    name_list_bis = []

    if labeled:
        name_list = [n for n in name_list if n.rfind('/') != len(n)-1 ]
        for n in name_list:
            count = n.count('/')
            if count > 1 or count == 0:
                raise HTTPException(status_code=422, detail='Format of zipfile not respected')
            idx = n.rfind('/')
            cls = n[: idx]
            name = n[idx + 1: n.rfind('.')] + '.png'

            name_list_bis.append((os.path.join(server.image_folder, cls), n))

            if not os.path.exists(os.path.join(server.image_folder, cls)):
                os.makedirs(os.path.join(server.image_folder, cls))
    else:
        for n in name_list:
            count = n.count('/')
            if count != 0:
                raise HTTPException(status_code=422, detail='Format of zipfile not respected')
            name = n[: n.rfind('.')] + '.png'
            name_list_bis.append((server.image_folder, n))

    batch_size = 128
    num_batches = len(name_list_bis) // batch_size
    with torch.no_grad():
        for i in range(num_batches+1):
            if i == num_batches:
                names = name_list_bis[i*batch_size:]
            else:
                names = name_list_bis[i*batch_size: (i + 1)*batch_size]
            async with await lock.gen_wlock():
                id = int(server.database.r.get('last_id_{}'.format('labeled' if labeled else 'unlabeled')).decode('utf-8'))
                data = dataset.AddDatasetList(id, names, server.name, not server.database.feat_extract)
                for name, n in names:
                    img = Image.open(BytesIO(zf.read(n)))
                    img.save(os.path.join(name, server.name + '_' + str(id)) + '.png', 'PNG')
                    id += 1
                loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)
                for batch, n in loader:
                    batch = batch.view(-1, 3, 224, 224).to(device=next(server.model.parameters()).device)

                    out = server.model(batch).cpu()

                    server.database.add(out.numpy(), list(n), labeled)

                    server.database.save()

    return

@app.get('/remove_image')
async def remove_image(name: str, public_key: str, private_key: str):
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    do_have = os.path.exists(name)
    if do_have is False:
        return
    else:
        async with await lock.gen_wlock():
            server.database.remove(name)

@app.get('/heartbeat')
def heartbeat():
    return {'nbr_images_labeled': server.database.index_labeled.ntotal,
            'nbr_images_unlabeled': server.database.index_unlabeled.ntotal}

class CytomineImage(BaseModel):
    id: int
    width: int
    height: int
    project: str
    resolution: str = None
    magnification: int = None
    filename: str
    originalFilename: str

@app.get('/add_slide')
async def add_slide(public_key: str, private_key: str, image_params: CytomineImage):
    loop = asyncio.get_running_loop()
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
        im = ImageInstance()
        im.id = image_params.id
        im.width, im.height = image_params.width, image_params.height
        im.resolution = image_params.resolution
        im.magnification = image_params.magnification
        im.filename, im.originalFilename = image_params.filename, image_params.originalFilename
        await loop.run_in_executor(
            server.pool, im.download, os.path.join(str(image_params.project), '{originalFilename}'))

    slide = OpenSlide(os.path.join(str(image_params.project), image_params.originalFilename))
    slide.get_thumbnail((512, 512))

    patch_x, patch_y = image_params.width // 224, image_params.height // 224
    thumb = slide.get_thumbnail((patch_x, patch_y)).convert('L')
    _, th = cv2.threshold(np.array(thumb), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    patches = np.transpose(np.where(th == 0))

    with torch.no_grad():
        data = AddSlide(patches, slide)
        outs = np.zeros((data.__len__(), server.model.num_features))

        data = torch.utils.data.DataLoader(data, 128, pin_memory=True)

        for i, image in enumerate(data):
            image = await loop.run_in_executor(
                server.pool, image.to, server.model.device)
            outs[128 * i: (i+1) * 128, :] = (await loop.run_in_executor(
                server.pool, server.model, image)).cpu().numpy()

    k = await loop.run_in_executor(server.pool, KMeans(500).fit, outs)

    name_list = []
    idx_list = []

    async with await lock.gen_wlock():
        id = int(server.database.r.get('last_id_unlabeled').decode('utf-8'))
        for i in range(500):
            idx = np.where(k.labels_ == i)[0]
            idx_tmp = np.absolute(outs[k.labels_ == i] - k.cluster_centers_[i]).sum(axis=1).argmin()
            idx_list.append(idx[idx_tmp])

            if idx[idx_tmp] >= patches.shape[0]:
                img = slide.read_region((patches[idx[idx_tmp]-patches.shape[0], 1] * 224,
                                         patches[idx[idx_tmp]-patches.shape[0], 0] * 224), 0, (224, 224)).convert('RGB')
            else:
                img = slide.read_region((patches[idx[idx_tmp], 1] * 224,
                                         patches[idx[idx_tmp], 0] * 224), 0, (224, 224)).convert('RGB')

            name = os.path.join(server.image_folder, '{}.png'.format(id))

            img.save(name, 'PNG')
            name_list.append(name)
            id += 1
        class ds(torch.utils.data.Dataset):
            def __init__(self, vectors, name_list, root):
                self.root = root
                self.vectors = vectors
                self.name_list = name_list
            def __len__(self):
                return len(self.name_list)
            def __getitem__(self, key):
                return self.vectors[key, :], os.path.join(self.name_list[key])

        with torch.no_grad():
            outs = outs[idx_list, :]
            data = ds(outs, name_list, server.image_folder)
            data = torch.utils.data.DataLoader(data, 128, pin_memory=True)
            for i, (vectors, names) in enumerate(data):
                server.database.add(vectors.numpy().astype(np.float32), names, False)
                server.database.save()

    return

@app.get('/add_slide_annotations')
async def add_slide_annotations(public_key: str, private_key: str, project_id: int, term: str):
    loop = asyncio.get_running_loop()
    with Cytomine(host=server.host, public_key=public_key, private_key=private_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorised')
        terms = TermCollection().fetch_with_filter('project', project_id)
        term_id = None
        for t in terms:
            if t.name == term:
                term_id = t.id
        if term_id is None:
            raise HTTPException(status_code=404, detail='Term {} not found in project with id {}'.format(term, project_id))
        annotations = AnnotationCollection()
        annotations.project = project_id
        annotations.showWKT = True
        annotations.showMeta = True
        annotations.showGIS = True
        annotations.showTerm = True
        annotations.fetch()

        if not os.path.exists(os.path.join(server.image_folder, term)):
            os.makedirs(os.path.join(server.image_folder, term))

        image_names = []

        for annotation in annotations:
            if term_id in annotation.term and annotation.area >= 700 and annotation.area < 5000:
                name = os.path.join(term, '{}.jpg'.format(annotation.id))
                await loop.run_in_executor(
                    server.pool, annotation.dump, os.path.join(server.image_folder, name))
                image_names.append(name)

    batch_size = 128
    data = dataset.AddDatasetList(server.image_folder, image_names, not server.database.feat_extract)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)
    with torch.no_grad():
        for batch, n in loader:
            async with await lock.gen_wlock():
                batch = batch.view(-1, 3, 224, 224).to(device=next(server.model.parameters()).device)

                out = server.model(batch).cpu()

                server.database.add(out.numpy(), list(n), True)
                server.database.save()
    return

if __name__ == '__main__':
    uvicorn.run('server:app', host=args.ip, port= args.port, reload=True,
                log_level='debug', workers=multiprocessing.cpu_count())
