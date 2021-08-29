from fastapi import FastAPI, File, UploadFile, Request, Cookie, HTTPException
import uvicorn
import argparse
import json
import aiohttp
import requests
import multiprocessing
from cytomine import Cytomine
from cytomine.models import CurrentUser
from io import BufferedReader, BytesIO
from collections import defaultdict
import numpy as np
import faiss
import asyncio
import random
import zipfile
from cytomine.models.image import ImageInstanceCollection
import threading
import time

app = FastAPI()

class Master:
    def __init__(self, ip, port, servers, http, host):
        self.ip = ip
        self.port = port

        if servers is not None:
            self.servers = {s: 0 for s in servers}
        else:
            self.servers = {}
        self.http = 'http' if http else 'https'
        self.host = host

        threading.Thread(target=self.heartbeat, daemon=True).start()

    def heartbeat(self):
        while True:
            for ip in self.servers.keys():
                try:
                    req = requests.get('{}://{}/heartbeat'.format(self.http, ip), timeout=5)

                except Exception as e:
                    self.servers[ip] = (np.inf, np.inf)
            time.sleep(5)



parser = argparse.ArgumentParser()

parser.add_argument(
    '--ip',
    default='127.0.0.1'
)

parser.add_argument(
    '--port',
    default=8000,
    type=int
)

parser.add_argument(
    '--server_addresses',
    nargs='+'
)

parser.add_argument(
    '--http',
    action='store_true'
)

parser.add_argument(
    '--host'
)

args = parser.parse_args()


if __name__ != '__main__':
    master = Master(args.ip, args.port, args.server_addresses, args.http, args.host)
    lock = defaultdict(lambda: asyncio.Lock())

def sort_ips(ip_list, labeled):
    if labeled:
        ip_list.sort(key = lambda k: k[1][0])
    else:
        ip_list.sort(key = lambda k: k[1][1])

    return [i for i, j in ip_list]

@app.get('/connect')
def connect(ip: str, nbr_images_labeled: int, nbr_images_unlabeled: int):
    master.servers[ip] = (nbr_images_labeled, nbr_images_unlabeled)
    return

@app.get('/servers_list')
def servers_list():
    return {'list': list(master.servers.items())}

@app.post('/get_nearest_images')
async def nearest_images(nrt_neigh: int, client_pub_key: str='', only_labeled: str='true',
                         client_pri_key: str='', image: UploadFile=File(...)):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')

    content = await image.read()
    responses = []
    ip_list = list(master.servers.keys())

    async with lock[client_pub_key]:
        async with aiohttp.ClientSession(trust_env=True) as session:
            for ip in ip_list:
                data = aiohttp.FormData()
                data.add_field('image', content, filename=image.filename, content_type='multipart/form-data')
                try:
                    async with session.post('{}://{}/nearest_neighbours/{}'.format(master.http, ip, only_labeled),
                                            data=data,
                                            params={'nrt_neigh': nrt_neigh, 'public_key': client_pub_key,
                                                    'private_key': client_pri_key},
                                            headers={'Content-Encoding': 'gzip'},
                                            timeout=aiohttp.ClientTimeout(100)) as resp:
                        responses.append(await resp.json())
                except Exception as e:
                    print(e)
                    responses.append(None)
        if np.all(np.array(responses) == None):
            raise HTTPException(status_code=500, detail='No server alive')

        indices = [i for i, _ in enumerate(responses) if responses[i] is not None]
        ip_list = [ip_list[i] for i in indices]
        req = [responses[i] for i in indices]

        distances = np.zeros((len(req) * nrt_neigh, 1), dtype=np.float32)
        for i, r in enumerate(req):
            distances[i * nrt_neigh: i * nrt_neigh + len(r['distances']), :] = np.array(r['distances']).reshape((-1, 1))
            distances[i * nrt_neigh + len(r['distances']): (i+1) * nrt_neigh, :] = 100000
        index = faiss.IndexFlatL2(1)
        index.add(distances)
        _, labels = index.search(np.array([[0]], dtype=np.float32), nrt_neigh)

        to_retrieve = defaultdict(list)

        for l in labels[0]:
            to_retrieve[l // nrt_neigh].append(int(l % nrt_neigh))

        images = []
        cls = []
        names = []
        async with aiohttp.ClientSession(trust_env=True) as session:
            for ip, lbls in zip(ip_list, to_retrieve.values()):
                if lbls != []:
                    try:
                        async with session.get('{}://{}/retrieve_images/{}'.format(master.http, ip, only_labeled),
                                               params={'public_key': client_pub_key, 'private_key': client_pri_key},
                                               json={'labels': list(lbls)},
                                               timeout=aiohttp.ClientTimeout(100)) as resp:
                            images.append((await resp.json())['images'])
                            cls.append((await resp.json())['cls'])
                            names.append((await resp.json())['names'])

                    except Exception as e:
                        pass
    if images == []:
        raise HTTPException(status_code=500, detail='No server alive')

    return {'images': [i for sublist in images for i in sublist],
            'cls': [c for sublist in cls for c in sublist],
            'names': [n for sublist in names for n in sublist],
            'distances': [float(distances[l, 0]) for l in list(labels[0])]}

@app.post('/index_image')
async def put_image(client_pub_key: str, client_pri_key: str, image: UploadFile=File(...), label: str=''):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
    content = await image.read()
    ip_list = sort_ips(list(master.servers.items()), label != '')
    while True:
        ip = ip_list.pop(0)

        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                data = aiohttp.FormData()
                data.add_field('image', content, filename=image.filename, content_type='multipart/form-data')
                async with session.post('{}://{}/index_image'.format(master.http, ip),
                                        data=data,
                                        params={'label': label, 'public_key': client_pub_key,
                                                'private_key': client_pri_key},
                                        headers={'Content-Encoding': 'gzip'},
                                        timeout=aiohttp.ClientTimeout(100)) as resp:
                    status = resp.status
                    if status == 409 or status == 422:
                        raise HTTPException(status_code=status, detail= await resp.json()['detail'])
                    if status != 200 and ip_list == []:
                        raise HTTPException(status_code=500, detail='No server alive')
                    if status == 200:
                        break
            except Exception as e:
                if ip_list == []:
                    raise HTTPException(status_code=500, detail='No server alive')

@app.post('/index_folder')
async def put_folder(client_pub_key: str, client_pri_key: str, labeled: bool, folder: UploadFile=File(...)):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
    content = await folder.read()

    ip_list = sort_ips(list(master.servers.items()), labeled)
    print(ip_list)
    while True:
        ip = ip_list.pop(0)
        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                data = aiohttp.FormData()
                data.add_field('folder', content, filename=folder.filename, content_type='multipart/form-data')
                async with session.post('{}://{}/index_folder'.format(master.http, ip),
                                        data=data,
                                        params={'labeled': str(labeled), 'public_key': client_pub_key,
                                                'private_key': client_pri_key},
                                        headers={'Content-Encoding': 'gzip'},
                                        timeout=aiohttp.ClientTimeout(5)) as resp:
                    status = resp.status
                    if status == 409 or status == 422 or status == 401:
                        raise HTTPException(status_code=status, detail= (await resp.json())['detail'])
                    if status != 200 and ip_list == []:
                        raise HTTPException(status_code=500, detail='No server alive')
                    if status == 200:
                        break
            except HTTPException as h:
                raise HTTPException(h.status_code, h.detail)
            except Exception as e:
                if ip_list == []:
                    raise HTTPException(status_code=500, detail='No server alive')

@app.get('/remove_image')
async def remove_image(client_pub_key: str, client_pri_key: str, name: str):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
    ip_list = list(master.servers.keys())

    for ip in ip_list:
        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                async with session.get('{}://{}/remove_image'.format(master.http, ip),
                                       params={'name': name, 'public_key': client_pub_key,
                                               'private_key': client_pri_key},
                                       timeout=aiohttp.ClientTimeout(5)) as resp:
                    pass
            except Exception as e:
                pass

@app.get('/index_slides')
async def add_slides(client_pub_key: str, client_pri_key: str, project_id: str):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
        image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)

    for image in image_instances:
        ip_list = sort_ips(list(master.servers.items()), False)
        while True:
            ip = ip_list.pop()
            async with aiohttp.ClientSession(trust_env=True) as session:
                try:
                    await session.get('{}://{}/add_slide'.format(master.http, ip),
                                      params={'public_key': client_pub_key,
                                              'private_key': client_pri_key},
                                      json={'id': image.id, 'width':image.width, 'project': project_id,
                                            'height': image.height, 'resolution': image.resolution,
                                            'magnification': image.magnification,
                                            'filename': image.filename, 'originalFilename': image.filename},
                                      timeout=aiohttp.ClientTimeout(300))
                    break
                except Exception as e:
                    print(e)
                    if ip_list == []:
                        raise HTTPException(status_code=500, detail='No server alive')



@app.get('/index_slide_annotations')
async def add_slides_annotations(client_pub_key: str, client_pri_key: str, project_id: str, label: str):
    with Cytomine(host=master.host, public_key=client_pub_key, private_key=client_pri_key):
        user = CurrentUser().fetch()
        if user is False:
            raise HTTPException(401, 'Unauthorized')
        image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)

    ip_list = sort_ips(list(master.servers.items()), True)
    while True:
        ip = ip_list.pop(0)
        async with aiohttp.ClientSession(trust_env=True) as session:
            try:
                await session.get('{}://{}/add_slide_annotations'.format(master.http, ip),
                                  params={'public_key': client_pub_key,
                                          'private_key': client_pri_key,
                                          'project_id': project_id,
                                          'term': label},
                                  timeout=aiohttp.ClientTimeout(1000))
                break
            except Exception as e:
                print(e)
                if ip_list == []:
                    raise HTTPException(status_code=500, detail='No server alive')


if __name__ == '__main__':
    uvicorn.run('master:app', host=args.ip, port=args.port, reload=True,
                debug=False, workers=multiprocessing.cpu_count())
