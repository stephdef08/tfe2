from fastapi import FastAPI, File, UploadFile, Request, Cookie, HTTPException
import uvicorn
import argparse
import json
import aiohttp
import multiprocessing
from cytomine import Cytomine
from io import BufferedReader, BytesIO
from collections import defaultdict
import numpy as np
import faiss
import asyncio
import random
import zipfile

app = FastAPI()

class Master:
    def __init__(self, ip, port, servers, http):
        self.ip = ip
        self.port = port
        self.servers = set(servers)
        self.http = 'http' if http else 'https'

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

args = parser.parse_args()


if __name__ != '__main__':
    master = Master(args.ip, args.port, args.server_addresses, args.http)
    lock = defaultdict(lambda: asyncio.Lock())

@app.get('/servers_list')
def servers_list():
    return {'list': list(master.servers)}

@app.post('/get_nearest_images')
async def nearest_images(nrt_neigh: int, client_pub_key: str='', only_labeled: str='true',
                         client_pri_key: str='', image: UploadFile=File(...)):
    with Cytomine(host='https://research.cytomine.be/', public_key=client_pub_key, private_key=client_pri_key):
        content = await image.read()
        responses = []
        ip_list = list(master.servers)
        async with lock[client_pub_key]:
            async with aiohttp.ClientSession(trust_env=True) as session:
                for ip in ip_list:
                    data = aiohttp.FormData()
                    data.add_field('image', content, filename=image.filename, content_type='multipart/form-data')
                    try:
                        async with session.post('{}://{}/nearest_neighbours/{}'.format(master.http, ip, only_labeled),
                                                data=data,
                                                params={'nrt_neigh': nrt_neigh, 'public_key': client_pub_key},
                                                headers={'Content-Encoding': 'gzip'},
                                                timeout=aiohttp.ClientTimeout(5)) as resp:
                            responses.append(await resp.json())
                    except Exception as e:
                        responses.append(None)
            if np.all(np.array(responses) == None):
                raise HTTPException(status_code=500, detail='No server alive')

            indices = [i for i, _ in enumerate(responses) if responses[i] is not None]
            ip_list = [ip_list[i] for i in indices]
            req = [responses[i] for i in indices]

            distances = np.zeros((len(req) * nrt_neigh, 1), dtype=np.float32)
            for i, r in enumerate(req):
                distances[i * nrt_neigh: i * nrt_neigh + len(r['distances']), :] = np.array(r['distances']).reshape((-1, 1))
                distances[i * nrt_neigh + len(r['distances']): (i+1) * nrt_neigh, :] = np.inf
            index = faiss.IndexFlatL2(1)
            index.add(distances)
            _, labels = index.search(np.array([[0]], dtype=np.float32), nrt_neigh)

            to_retrieve = defaultdict(list)

            for l in labels[0]:
                to_retrieve[l // nrt_neigh].append(int(l % nrt_neigh))

            images = []
            cls = []
            async with aiohttp.ClientSession(trust_env=True) as session:
                for ip, labels in zip(ip_list, to_retrieve.values()):
                    if labels != []:
                        try:
                            async with session.get('{}://{}/retrieve_images/{}'.format(master.http, ip, only_labeled),
                                                   params={'public_key': client_pub_key},
                                                   json={'labels': list(labels)},
                                                   timeout=aiohttp.ClientTimeout(5)) as resp:
                                images.append((await resp.json())['images'])
                                cls.append((await resp.json())['cls'])

                        except Exception as e:
                            pass

    if images == []:
        raise HTTPException(status_code=500, detail='No server alive')

    return {'images': [[i for i in im] for im in images],
            'cls': [[c for c in cl] for cl in cls]}

@app.post('/put_image')
async def put_image(client_pub_key: str, client_pri_key: str, image: UploadFile=File(...), label: str=''):
    with Cytomine(host='https://research.cytomine.be/', public_key=client_pub_key, private_key=client_pri_key):
        content = await image.read()

        ip_list = list(master.servers)

        while True:
            ip = random.choice(ip_list)
            ip_list.remove(ip)
            async with aiohttp.ClientSession(trust_env=True) as session:
                try:
                    data = aiohttp.FormData()
                    data.add_field('image', content, filename=image.filename, content_type='multipart/form-data')
                    async with session.post('{}://{}/index_image'.format(master.http, ip),
                                            data=data,
                                            params={'label': label},
                                            headers={'Content-Encoding': 'gzip'},
                                            timeout=aiohttp.ClientTimeout(5)) as resp:
                        status = resp.status
                        if status != 200 and ip_list == []:
                            raise HTTPException(status_code=500, detail='No server alive')
                        break
                except Exception as e:
                    if ip_list == []:
                        raise HTTPException(status_code=500, detail='No server alive')

@app.post('/put_folder')
async def put_folder(client_pub_key: str, client_pri_key: str, folder: UploadFile=File(...)):
    content = await folder.read()

    zf = zipfile.ZipFile(BytesIO(content))
    name_list = zf.name_list()

@app.post('/remove_image')
async def remove_image(client_pub_key: str, client_pri_key: str, name: str):
    with Cytomine(host='https://research.cytomine.be/', public_key=client_pub_key, private_key=client_pri_key):
        ip_list = list(master.servers)

        for ip in ip_list:
            async with aiohttp.ClientSession(trust_env=True) as session:
                try:
                    async with session.get('{}://{}/remove_image'.format(master.http, ip),
                                           params={'name': name},
                                           timeout=aiohttp.ClientTimeout(5)) as resp:
                        pass
                except Exception as e:
                    pass



if __name__ == '__main__':
    uvicorn.run('master:app', host=args.ip, port=args.port, reload=True,
                debug=False, workers=multiprocessing.cpu_count())
