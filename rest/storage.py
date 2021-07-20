import requests
from PIL import Image
from io import BytesIO
from joblib import Parallel, delayed
import multiprocessing
import json
import faiss
import numpy as np
from collections import defaultdict

class RestError:
    pass

class Storage:
    def __init__(self, ip, port, http=True, nrt_neigh=10):
        self.ip = ip
        self.port = port
        self.http = 'http' if http else https
        self.thread_pool = Parallel(n_jobs=multiprocessing.cpu_count(),
                                    prefer="threads")
        self.nrt_neigh = nrt_neigh

    def get(self, path):
        req = requests.get('{}://{}:{}/'.format(self.http, self.ip, self.port),
                           params='servers_list')

        if req.status_code != 200:
            raise RestError('get {} {}'.format(path, req.status_code))

        img = Image.open(path).convert('RGB')
        bytes_io = BytesIO()
        img.save(bytes_io, 'png')
        bytes_io.seek(0)

        ip_list = req.json()['list']

        req = self.thread_pool(delayed(requests.post)('{}://{}/nearest_neighbours'.format(self.http, address),
                                                      files={'file': (path, bytes_io, 'image/png')},
                                                      params={'nrt_neigh': self.nrt_neigh})
                               for address in ip_list)

        distances = np.zeros((len(req) * self.nrt_neigh, 1), dtype=np.float32)
        for i, r in enumerate(req):
            distances[i * self.nrt_neigh: (i+1) * self.nrt_neigh, :] = np.array(r.json()['distances']).reshape((self.nrt_neigh, 1))
        index = faiss.IndexFlatL2(1)
        index.add(distances)
        _, labels = index.search(np.array([[0]], dtype=np.float32), self.nrt_neigh)

        to_retrieve = defaultdict(list)

        for i in range(len(ip_list)):
            to_retrieve[i]

        for l in labels[0]:
            to_retrieve[l // self.nrt_neigh].append(l % self.nrt_neigh)

        req = self.thread_pool(delayed(requests.get)('{}://{}/retrieve_images'.format(self.http, address),
                                                      params={'labels': list(labels)})
                               for address, labels in zip(ip_list, to_retrieve.values()))

        for r in req:
            for img, s in zip(r.json()['images'], r.json()['sizes']):
                print(s)
                Image.frombytes('RGB',  BytesIO(bytes(img, 'ascii'))).show()


store = Storage('127.0.0.1', 8000)

store.get('/home/stephan/Documents/tfe1/patch/val/warwick_crc1/116153472_0_100_100_100.png')
