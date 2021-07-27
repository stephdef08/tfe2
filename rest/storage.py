import requests
from PIL import Image
from io import BytesIO
from joblib import Parallel, delayed
import multiprocessing
import json
import faiss
import numpy as np
from collections import defaultdict
import base64
from socket import error as socket_error
import random
import zipfile
import os

class ApiError(Exception):
    pass

class Storage:
    def __init__(self, ip, port, public_key, http=True, nrt_neigh=10):
        self.ip = ip
        self.port = port
        self.http = 'http' if http else https
        self.thread_pool = Parallel(n_jobs=multiprocessing.cpu_count(),
                                    prefer="threads")
        self.nrt_neigh = nrt_neigh
        self.public_key = public_key

    def get(self, path):
        try:
            req = requests.get('{}://{}:{}/servers_list'.format(self.http, self.ip, self.port),
                               timeout=3)
        except socket_error:
            raise ApiError('Error connecting to master')

        if req.status_code != 200:
            raise ApiError('get {} {}'.format(path, req.status_code))

        ip_list = req.json()['list']

        def post(url, files, params, headers):
            try:
                req = requests.post(url, files=files, params=params, headers=headers,
                                    timeout=3)
                return req
            except socket_error:
                return None

        req = self.thread_pool(delayed(post)('{}://{}/nearest_neighbours'.format(self.http, address),
                                             files={'image': (path, open(path, 'rb'), 'multipart/form-data')},
                                             params={'nrt_neigh': self.nrt_neigh, 'public_key': self.public_key},
                                             headers={'Content-Encoding': 'gzip'})
                               for address in ip_list)

        if np.all(np.array(req) == None):
            raise ApiError('No server alive')

        indices = [i for i, _ in enumerate(req) if req[i] is not None]
        ip_list = [ip_list[i] for i in indices]
        req = [req[i] for i in indices]

        distances = np.zeros((len(req) * self.nrt_neigh, 1), dtype=np.float32)
        for i, r in enumerate(req):
            distances[i * self.nrt_neigh: (i+1) * self.nrt_neigh, :] = np.array(r.json()['distances']).reshape((self.nrt_neigh, 1))
        index = faiss.IndexFlatL2(1)
        index.add(distances)
        _, labels = index.search(np.array([[0]], dtype=np.float32), self.nrt_neigh)

        to_retrieve = defaultdict(list)

        for l in labels[0]:
            to_retrieve[l // self.nrt_neigh].append(int(l % self.nrt_neigh))

        def req_get(urls, params, json):
            try:
                req = requests.get(urls, params=params, json=json, timeout=3)
                return req
            except socket_error:
                return None

        req = self.thread_pool(delayed(req_get)('{}://{}/retrieve_images'.format(self.http, address),
                                                params={'public_key': self.public_key},
                                                json={'labels': list(labels)})
                               for address, labels in zip(ip_list, to_retrieve.values()) if labels != [])

        if np.all(np.array(req) == None):
            raise ApiError('No server alive')

        for r in req:
            if r is not None:
                for a in r.json()['images']:
                    Image.open(BytesIO(base64.b64decode(a))).show()

    def put(self, path):
        req = requests.get('{}://{}:{}/servers_list'.format(self.http, self.ip, self.port))

        if req.status_code != 200:
            raise ApiError('get {} {}'.format(path, req.status_code))

        ip_list = req.json()['list']

        while True:
            ip = random.choice(ip_list)
            ip_list.remove(ip)

            try:
                req = requests.post('{}://{}/index_image'.format(self.http, ip),
                                    files={'image': (path, open(path, 'rb'), 'multipart/form-data')},
                                    headers={'Content-Encoding': 'gzip'})
            except socket_error:
                if ip_list != []:
                    pass
                else:
                    raise ApiError('No server alive')
            else:
                if req.status_code != 200:
                    raise ApiError(req.json()['detail'])
                break

    def put_folder(self, path):
        req = requests.get('{}://{}:{}/servers_list'.format(self.http, self.ip, self.port))

        if req.status_code != 200:
            raise ApiError('get {} {}'.format(path, req.status_code))

        ip_list = req.json()['list']

        zipf = zipfile.ZipFile('archive.zip', 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(path, file))
        zipf.close()

        while True:
            ip = random.choice(ip_list)
            ip_list.remove(ip)

            try:
                req = requests.post('{}://{}/index_folder'.format(self.http, ip),
                                    files={'folder': ('archive.zip', open('archive.zip', 'rb'), 'multipart/form-data')},
                                    headers={'Content-Encoding': 'gzip'})
            except socket_error:
                if ip_list != []:
                    pass
                else:
                    raise ApiError('No server alive')
            else:
                if req.status_code != 200:
                    raise ApiError()
                break

        os.remove('archive.zip')
