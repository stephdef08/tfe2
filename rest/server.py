from fastapi import FastAPI, Request, File, UploadFile, HTTPException
import uvicorn
from PIL import Image
from fastapi.responses import StreamingResponse
from io import BytesIO
from io import StringIO
import argparse
from database.db import Database
from database.densenet import Model
from database import transformer
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

app = FastAPI()

class Server:
    def __init__(self, ip, port, master_ip, master_port, model, num_features,
                 weights, use_dr, device, attention, image_folder):
        self.ip = ip
        self.port = port
        self.master_ip = master_ip
        self.master_port = master_port
        if model != 'transformer':
            self.model = Model(model, num_features=num_features, name=weights,
                               use_dr=use_dr, device=device, attention=attention)
            self.feat_extract = None
        else:
            self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                     size=224, do_center_crop=False,
                                                                     image_mean=[0.485, 0.456, 0.406],
                                                                     image_std=[0.229, 0.224, 0.225])
            self.model = transformer.Model(num_features, name=weights, device=device)
        self.database = Database("db", self.model, load=True)
        self.image_folder = image_folder

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
    '--model',
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
    '--attention',
    action='store_true'
)

parser.add_argument(
    '--folder',
    default='images'
)

args = parser.parse_args()

if args.gpu_id >= 0:
    device = 'cuda:' + str(args.gpu_id)
else:
    device = 'cpu'

if __name__ != '__main__':
    server = Server(args.ip, args.port, args.master_ip, args.master_port,
                    args.model, args.num_features, args.weights, args.use_dr,
                    device, args.attention, args.folder)

    names = {}

    lock = asyncio.Lock()

@app.post('/nearest_neighbours/{retrieve_class}')
async def nearest_neighbours(nrt_neigh: int, public_key: str, retrieve_class: str, image: UploadFile=File(...)):
    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB')

    async with lock:
        names[public_key], distance, _, _ = server.database.search(img, nrt_neigh, retrieve_class)

    return {'distances': distance[0].tolist()}

class LabelList(BaseModel):
    labels: list

@app.get('/retrieve_images/{retrieve_class}')
def retrieve_images(retrieve_class: str, labels: LabelList, public_key: str=''):
    images = []

    if retrieve_class == 'true':
        cls = []

        for i in labels.labels:
            img = Image.open(names[public_key][int(i)])
            end = names[public_key][int(i)].rfind("/")
            begin = names[public_key][int(i)].rfind("/", 0, end) + 1
            cls.append(names[public_key][int(i)][begin: end])
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))

        return {'images': images, 'cls': cls}
    elif retrieve_class == 'false':
        for i in labels.labels:
            img = Image.open(names[public_key][int(i)])
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))
        return {'images': images, 'cls': ["Unkown" for i in images]}
    elif retrieve_class == 'mix':
        cls = []

        for i in labels.labels:
            img = Image.open(names[public_key][int(i)])
            if names[public_key].count('/') > 1:
                end = names[public_key][int(i)].rfind("/")
                begin = names[public_key][int(i)].rfind("/", 0, end) + 1
                cls.append(names[public_key][int(i)][begin: end])
            else:
                cls.append('Unkown')
            bytes_io = BytesIO()
            img.save(bytes_io, 'png')
            bytes_io.seek(0)

            images.append(base64.b64encode(bytes_io.getvalue()))

        return {'images': images, 'cls': cls}

@app.post('/index_image')
async def index_image(image: UploadFile=File(...), label: str=''):
    content = await image.read()
    img = Image.open(BytesIO(content)).convert('RGB')
    name = image.filename[image.filename.rfind('/')+1: image.filename.rfind('.')] + '.png'

    if os.path.exists(os.path.join(server.image_folder, label, name)):
        raise HTTPException(status_code=409, detail='File already exists')

    if not os.path.exists(os.path.join(server.image_folder, label)):
        os.makedirs(os.path.join(server.image_folder, label))
    img.save(os.path.join(server.image_folder, label, name), 'PNG')

    async with lock:
        with torch.no_grad():
            if not server.feat_extract:
                image = transforms.Resize((224, 224))(img)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(image)
            else:
                image = server.feat_extract(images=img, return_tensors='pt')['pixel_values']

            out = server.model(image.to(device=next(server.model.parameters()).device).view(1, 3, 224, 224))

        server.database.add(out.cpu().numpy().reshape(-1, server.model.num_features),
                            [os.path.join(server.image_folder, label, name)], label!='')

        server.database.save()
    return

@app.post('/index_folder')
async def index_folder(folder: UploadFile=File(...)):
    content = await folder.read()

    zf = zipfile.ZipFile(BytesIO(content))
    name_list = zf.namelist()

    name_list_bis = []

    for n in name_list:
        name = n[n.rfind('/')+1: n.rfind('.')] + '.png'
        name_list_bis.append(name)

        if not os.path.exists(os.path.join(server.image_folder, n)):
            img = Image.open(BytesIO(zf.read(n)))
            img.save(os.path.join(server.image_folder, name), 'PNG')

    batch_size = 128
    data = dataset.AddDatasetList(server.image_folder, name_list_bis, not server.feat_extract)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, pin_memory=True)
    with torch.no_grad():
        for batch, n in loader:
            async with lock:
                batch = batch.view(-1, 3, 224, 224).to(device=next(server.model.parameters()).device)

                out = server.model(batch).cpu()

                server.database.add(out.numpy(), list(n))

    server.database.save()

    return

@app.get('/remove_image')
async def remove_image(name: str):
    do_have = os.path.exists(name)
    if do_have is False:
        return
    else:
        async with lock:
            server.database.remove(name)

if __name__ == '__main__':
    uvicorn.run('server:app', host=args.ip, port= args.port, reload=True,
                log_level='debug', workers=multiprocessing.cpu_count())
