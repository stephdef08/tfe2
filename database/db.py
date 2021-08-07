import faiss
import database.densenet as densenet
import torch
import database.dataset as dataset
from PIL import Image
from torchvision import transforms
import redis
import numpy as np
from transformers import ViTFeatureExtractor
from transformers import DeiTFeatureExtractor
import time
import json
import os

class Database:
    def __init__(self, filename, model, load=False, transformer=False, device='gpu'):
        self.name = filename
        self.embedding_size = 128
        self.model = model
        self.device = device

        res_labeled = faiss.StandardGpuResources()
        res_unlabeled = faiss.StandardGpuResources()

        if load == True:
            self.index_labeled = faiss.read_index(filename + '_labeled')
            self.index_unlabeled = faiss.read_index(filename + '_unlabeled')
            self.r = redis.Redis(host='localhost', port='6379', db=0)
        else:
            self.index_labeled = faiss.IndexFlatL2(self.embedding_size)
            self.index_labeled = faiss.IndexIDMap(self.index_labeled)
            self.index_unlabeled = faiss.IndexFlatL2(self.embedding_size)
            self.index_unlabeled = faiss.IndexIDMap(self.index_unlabeled)
            self.r = redis.Redis(host='localhost', port='6379', db=0)
            self.r.flushdb()

            self.r.set('last_id_labeled', 0)
            self.r.set('last_id_unlabeled', 0)

        if device == 'gpu':
            self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
            self.index_unlabeled = faiss.index_cpu_to_gpu(res_unlabeled, 0, self.index_unlabeled)

        self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                 size=224, do_center_crop=False,
                                                                 image_mean=[0.485, 0.456, 0.406],
                                                                 image_std=[0.229, 0.224, 0.225]) if transformer else None

    def add(self, x, names, label):
        if label:
            last_id = int(self.r.get('last_id_labeled').decode('utf-8'))
            self.index_labeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            for n, x_ in zip(names, x):
                self.r.set(str(last_id) + "labeled", '["'+n+'","'+str(x_)+'"]')
                self.r.set(n, str(last_id) + "labeled")
                last_id += 1
            self.r.set('last_id_labeled', last_id + x.shape[0])
        else:
            last_id = int(self.r.get('last_id_unlabeled').decode('utf-8'))
            self.index_unlabeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            for n, x_ in zip(names, x):
                self.r.set(str(last_id) + "unlabeled", '["'+n+'","'+str(x_)+'"]')
                self.r.set(n, str(last_id) + "unlabeled")
                last_id += 1
            self.r.set('last_id_unlabeled', last_id + x.shape[0])

    @torch.no_grad()
    def add_dataset(self, data_root, name_list=[], label=True):
        transformer = not self.feat_extract

        if name_list == []:
            data = dataset.AddDataset(data_root, transformer)
        else:
            data = dataset.AddDatasetList(data_root, name_list, transformer)
        loader = torch.utils.data.DataLoader(data, batch_size=128, num_workers=12, pin_memory=True)

        for i, (images, filenames) in enumerate(loader):
            images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)

            out = self.model(images).cpu()

            self.add(out.numpy(), list(filenames), label)

        self.save()

    @torch.no_grad()
    def search(self, x, nrt_neigh=10, retrieve_class='true'):
        t_model = time.time()
        if not self.feat_extract:
            image = transforms.Resize((224, 224))(x)
            image = transforms.ToTensor()(image)
            image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(image)
        else:
            image = self.feat_extract(images=x, return_tensors='pt')['pixel_values']

        out = self.model(image.to(device=next(self.model.parameters()).device).view(1, 3, 224, 224))
        t_model = time.time() - t_model
        t_search = time.time()

        if retrieve_class == 'true':
            distance, labels = self.index_labeled.search(out.cpu().numpy(), nrt_neigh)
            labels = [l for l in list(labels[0]) if l != -1]

            names = []
            for l in labels:
                n = self.r.get(str(l) + 'labeled').decode('utf-8')
                n = n[2:n.find(',')-1]
                names.append(n)
            t_search = time.time() - t_search

            return names, distance, t_model, t_search
        elif retrieve_class == 'false':
            distance, labels = self.index_unlabeled.search(out.cpu().numpy(), nrt_neigh)
            labels = [l for l in list(labels[0]) if l != -1]

            names = []
            for l in labels:
                n = self.r.get(str(l) + 'unlabeled').decode('utf-8')
                n = n[2:n.find(',')-1]
                names.append(n)
            t_search = time.time() - t_search

            return names, distance, t_model, t_search
        elif retrieve_class == 'mix':
            distance_l, labels_l = self.index_labeled.search(out.cpu().numpy(), nrt_neigh)
            distance_u, labels_u = self.index_unlabeled.search(out.cpu().numpy(), nrt_neigh)
            labels_l = [l for l in list(labels_l[0]) if l != -1]
            labels_u = [l for l in list(labels_u[0]) if l != -1]

            index = faiss.IndexFlatL2(1)
            index.add(np.array(distance_l, dtype=np.float32).reshape(-1, 1))
            index.add(np.array(distance_u, dtype=np.float32).reshape(-1, 1))

            _, labels = index.search(np.array([[0]], dtype=np.float32), nrt_neigh)

            names = []
            distance = []
            labels = [l for l in list(labels[0]) if l != -1]
            for l in labels:
                if l < nrt_neigh:
                    if l < len(labels_l):
                        n = self.r.get(str(labels_l[l]) + 'labeled').decode('utf-8')
                        distance.append(distance_l[0][l])
                        n = n[2:n.find(',')-1]
                        names.append(n)
                else:
                    if l < len(labels_u) + nrt_neigh:
                        n = self.r.get(str(labels_u[l - nrt_neigh]) + 'unlabeled').decode('utf-8')
                        distance.append(distance_u[0][l - nrt_neigh])
                        n = n[2:n.find(',')-1]
                        names.append(n)
            t_search = time.time() - t_search

            return names, np.array(distance).reshape(1, -1), t_model, t_search

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')

        labeled = key.find('unlabeled') == -1
        if labeled:
            idx = key.find('labeled')
        else:
            idx = key.find('unlabeled')
        label = int(key[:idx])

        idsel = faiss.IDSelectorRange(label, label+1)

        if labeled:
            if self.device == 'gpu':
                self.index_labeled = faiss.index_gpu_to_cpu(self.index_labeled)
            self.index_labeled.remove_ids(idsel)
            self.save()
            self.r.delete(key + 'labeled')
            self.r.delete(name)
            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
        else:
            if self.device == 'gpu':
                self.index_unlabeled = faiss.index_gpu_to_cpu(self.index_unlabeled)
            self.index_unlabeled.remove_ids(idsel)
            self.save()
            self.r.delete(key + 'unlabeled')
            self.r.delete(name)
            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_unlabeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_unlabeled)

        os.remove(name)

    def train(self, data_root):
        batch_size = 128

        class ds(torch.utils.data.Dataset):
            def __init__(self, r, labeled):
                self.r = r
                self.size = self.r.execute_command("DBSIZE")
                self.labeled = labeled

            def __len__(self):
                return self.size

            def __getitem__(self, key):
                vec = json.loads(self.r.get(str(key) + self.labeled).decode("utf-8"), strict=False)[1]
                vec = vec.replace('\n', '').replace('[', '').replace(']', '')
                vec = np.fromstring(vec, dtype=np.float32, sep=' ')
                return vec

        data = ds(self.r, 'labeled')
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             num_workers=12, pin_memory=True)

        x = np.zeros((data.__len__(), self.model.num_features), dtype=np.float32)

        for i, x_ in enumerate(loader):
            x[i * batch_size : (i+1) * batch_size, :] = x_

        self.quantizer = faiss.IndexFlatL2(self.model.num_features)
        self.index_labeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                        int(np.sqrt(data.__len__())))

        res_labeled = faiss.StandardGpuResources()
        self.index_labeled = faiss.index_cpu_to_gpu(res, 0, self.index_labeled)

        self.index_labeled.train(x)
        self.index_labeled.add(x)

        data = ds(self.r, 'unlabeled')
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             num_workers=12, pin_memory=True)

        x = np.zeros((data.__len__(), self.model.num_features), dtype=np.float32)

        for i, x_ in enumerate(loader):
            x[i * batch_size : (i+1) * batch_size, :] = x_

        self.quantizer = faiss.IndexFlatL2(self.model.num_features)
        self.index_unlabeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                        int(np.sqrt(data.__len__())))

        res_unlabeled = faiss.StandardGpuResources()
        self.index_unlabeled = faiss.index_cpu_to_gpu(res, 0, self.index_unlabeled)

        self.index_unlabeled.train(x)
        self.index_unlabeled.add(x)

    def save(self):
        if self.device != 'gpu':
            faiss.write_index(self.index_labeled, self.name + '_labeled')
            faiss.write_index(self.index_unlabeled, self.name + '_unlabeled')
        else:
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_labeled), self.name + '_labeled')
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_unlabeled), self.name + '_unlabeled')

if __name__ == "__main__":
    model = densenet.Model(num_features=128, name='weights/weights_IN_densenet_20')
    database = Database("db", model, load=True)
    print(database.index_labeled.ntotal)
    database.remove('../tfe1/patch/test/ulg_lbtd_lba2/75996_472563.png')
    database.remove('../tfe1/patch/test/ulb_anapath_lba6/377614_715383.png')
    database.remove('../tfe1/patch/test/janowczyk2_0/9023_116894917_600_100_200_200.png')
    print(database.index_labeled.ntotal)
