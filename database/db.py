import faiss
import database.models as models
import torch
import database.dataset as dataset
from PIL import Image
from torchvision import transforms
import redis
import numpy as np
from transformers import DeiTFeatureExtractor
import time
import json
import os
import argparse

class Database:
    def __init__(self, filename, model, load=False, transformer=False, device='cpu'):
        self.name = filename
        self.embedding_size = 128
        self.model = model
        self.device = device
        self.filename = filename

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

            open(filename + '_labeledvectors', 'w').close()
            open(filename + '_unlabeledvectors', 'w').close()

        if device == 'gpu':
            self.index_labeled = faiss.index_cpu_to_gpu(res_labeled, 0, self.index_labeled)
            self.index_unlabeled = faiss.index_cpu_to_gpu(res_unlabeled, 0, self.index_unlabeled)

        self.transformer = transformer
        self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                 size=224, do_center_crop=False,
                                                                 image_mean=[0.485, 0.456, 0.406],
                                                                 image_std=[0.229, 0.224, 0.225]) if transformer else None

    def add(self, x, names, label):
        if label:
            last_id = int(self.r.get('last_id_labeled').decode('utf-8'))
            self.index_labeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            with open(self.filename + '_labeledvectors', 'a') as file:
                for n, x_ in zip(names, x):
                    self.r.set(str(last_id) + 'labeled', n)
                    self.r.set(n, str(last_id) + 'labeled')
                    file.write('\n' + str(last_id) + str(x_))
                    last_id += 1

            self.r.set('last_id_labeled', last_id)
        else:
            last_id = int(self.r.get('last_id_unlabeled').decode('utf-8'))
            self.index_unlabeled.add_with_ids(x, np.arange(last_id, last_id + x.shape[0]))

            with open(self.filename + '_unlabeledvectors', 'a') as file:
                for n, x_ in zip(names, x):
                    self.r.set(str(last_id) + 'unlabeled', n)
                    self.r.set(n, str(last_id) + 'unlabeled')
                    file.write('\n' + str(last_id) + str(x_))
                    last_id += 1

            self.r.set('last_id_unlabeled', last_id)

    @torch.no_grad()
    def add_dataset(self, data_root, name_list=[], label=True):
        if name_list == []:
            data = dataset.AddDataset(data_root, self.transformer)
        else:
            data = dataset.AddDatasetList(data_root, name_list, self.transformer)
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

        out = self.model(image.to(device=next(self.model.parameters()).device).view(-1, 3, 224, 224))
        t_model = time.time() - t_model
        t_search = time.time()

        if retrieve_class == 'true':
            distance, labels = self.index_labeled.search(out.cpu().numpy(), nrt_neigh)

            labels = [l for l in list(labels[0]) if l != -1]

            names = []
            for l in labels:
                n = self.r.get(str(l) + 'labeled').decode('utf-8')
                names.append(n)
            t_search = time.time() - t_search

            return names, distance.tolist(), t_model, t_search
        elif retrieve_class == 'false':
            distance, labels = self.index_unlabeled.search(out.cpu().numpy(), nrt_neigh)
            labels = [l for l in list(labels[0]) if l != -1]
            names = []
            print(distance)
            for l in labels:
                n = self.r.get(str(l) + 'unlabeled').decode('utf-8')
                names.append(n)
            t_search = time.time() - t_search

            return names, distance.tolist(), t_model, t_search
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
                        names.append(n)
                else:
                    if l < len(labels_u) + nrt_neigh:
                        n = self.r.get(str(labels_u[l - nrt_neigh]) + 'unlabeled').decode('utf-8')
                        distance.append(distance_u[0][l - nrt_neigh])
                        names.append(n)
            t_search = time.time() - t_search

            return names, np.array(distance).reshape(1, -1).tolist(), t_model, t_search

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')

        labeled = key.find('unlabeled') == -1
        if labeled:
            idx = key.find('labeled')
        else:
            idx = key.find('unlabeled')

        try:
            label = int(key[:idx])
        except:
            pass

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

    def train_labeled(self):
        batch_size = 128
        x = []
        keys = []
        with open(self.filename + '_labeledvectors', 'r') as file:
            with open(self.filename + '_newlabeledvectors', 'w') as newfile:
                lines = file.readlines()
                str_num = ''
                for line in lines:
                    line = line.replace('\n', '')
                    idx = line.find('[')
                    if idx != -1:
                        nbr = int(line[:idx])
                        str_num += line[idx+1:]
                    else:
                        idx = line.find(']')
                        if idx != -1:
                            str_num += line[:idx]
                            if self.r.get(str(nbr) + 'labeled') is not None:
                                vec = np.fromstring(str_num, dtype=np.float32, sep=' ')
                                newfile.write('\n' + str(nbr) + str(vec))
                                keys.append(nbr)
                                x.append(vec)
                                str_num = ''
                        else:
                            str_num += line
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index_labeled.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index_labeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                    num_clusters)

            if self.device == 'gpu':
                res_labeled = faiss.StandardGpuResources()
                self.index_labeled = faiss.index_cpu_to_gpu(res, 0, self.index_labeled)

            x = np.array(x, dtype=np.float32)

            self.index_labeled.train(x)
            self.index_labeled.nprobe = num_clusters // 10

            num_batches = self.index_labeled.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index_labeled.add_with_ids(x_, np.array(key, dtype=np.int64))
        os.replace(self.filename + '_newlabeledvectors', self.filename + '_labeledvectors')

    def train_unlabeled(self):
        batch_size = 128
        x = []
        keys = []
        with open(self.filename + '_unlabeledvectors', 'r') as file:
            with open(self.filename + '_newunlabeledvectors', 'w') as newfile:
                lines = file.readlines()
                str_num = ''
                for line in lines:
                    line = line.replace('\n', '')
                    idx = line.find('[')
                    if idx != -1:
                        nbr = int(line[:idx])
                        str_num += line[idx+1:]
                    else:
                        idx = line.find(']')
                        if idx != -1:
                            str_num += line[:idx]
                            if self.r.get(str(nbr) + 'unlabeled') is not None:
                                vec = np.fromstring(str_num, dtype=np.float32, sep=' ')
                                newfile.write('\n' + str(nbr) + str(vec))
                                keys.append(nbr)
                                x.append(vec)
                                str_num = ''
                        else:
                            str_num += line
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index_unlabeled.ntotal))
            self.quantizer = faiss.IndexFlatL2(self.model.num_features)
            self.index_unlabeled = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                                      num_clusters)

            if self.device == 'gpu':
                res_unlabeled = faiss.StandardGpuResources()
                self.index_unlabeled = faiss.index_cpu_to_gpu(res, 0, self.index_unlabeled)

            x = np.array(x, dtype=np.float32)

            self.index_unlabeled.train(x)
            self.index_unlabeled.nprobe = num_clusters // 10

            num_batches = self.index_unlabeled.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index_unlabeled.add_with_ids(x_, np.array(key, dtype=np.int64))

        os.replace(self.filename + '_newunlabeledvectors', self.filename + '_unlabeledvectors')

    def save(self):
        if self.device != 'gpu':
            faiss.write_index(self.index_labeled, self.name + '_labeled')
            faiss.write_index(self.index_unlabeled, self.name + '_unlabeled')
        else:
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_labeled), self.name + '_labeled')
            faiss.write_index(faiss.index_gpu_to_cpu(self.index_unlabeled), self.name + '_unlabeled')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--extractor',
        default='densenet'
    )

    parser.add_argument(
        '--weights'
    )

    parser.add_argument(
        '--db_name'
    )

    parser.add_argument(
        '--unlabeled',
        action='store_true'
    )

    args = parser.parse_args()

    model = models.Model(num_features=128, model=args.extractor, name=args.weights)
    database = Database(args.db_name, model, load=True)
    if args.unlabeled:
        database.train_unlabeled()
        print(database.search(Image.open('/home/stephan/Documents/tfe4/1.png').convert('RGB'), 10, 'false'))
    else:
        database.train_labeled()
    database.save()
