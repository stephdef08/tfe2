import faiss
import csv
import database.densenet as densenet
import torch
import database.dataset as dataset
from PIL import Image
from torchvision import transforms
import redis
import numpy as np
from transformers import ViTFeatureExtractor
from transformers import DeiTFeatureExtractor

class Database:
    def __init__(self, filename, model, load=False, transformer=False):
        self.name = filename
        self.embedding_size = 128
        self.model = model

        res = faiss.StandardGpuResources()

        if load == True:
            self.index = faiss.read_index(filename)
            self.r = redis.Redis(host='localhost', port='6379', db=0)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_size)
            self.r = redis.Redis(host='localhost', port='6379', db=0)
            self.r.flushdb()

        self.count = self.index.ntotal

        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # self.feat_extract = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224',
        #                                                         image_mean=[0.485, 0.456, 0.406],
        #                                                         image_std=[0.229, 0.224, 0.225]) if not transformer else None
        self.feat_extract = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                 size=224, do_center_crop=False,
                                                                 image_mean=[0.485, 0.456, 0.406],
                                                                 image_std=[0.229, 0.224, 0.225]) if not transformer else None

    @torch.no_grad()
    def add(self, x, names):
        self.index.add(x)

        for n in names:
            self.r.set(str(self.count), n)
            self.count += 1

    @torch.no_grad()
    def add_dataset(self, data_root):
        # if not self.index.is_trained:
        #     print("Index not trained")
        #     exit()

        transformer = not self.feat_extract

        data = dataset.AddDataset(data_root, transformer)
        loader = torch.utils.data.DataLoader(data, batch_size=128,
                                             num_workers=12, pin_memory=True)

        for i, (images, filenames) in enumerate(loader):
            images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)

            out = self.model(images).cpu()

            self.add(out.numpy(), list(filenames))

        faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.name)

    @torch.no_grad()
    def search(self, x, nrt_neigh=10):
        # if not self.index.is_trained:
        #     print("Index not trained")
        #     exit()
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

        distance, labels = self.index.search(out.cpu().numpy(), nrt_neigh)

        names = []

        for l in labels[0]:
            names.append(self.r.get(str(l)).decode("utf-8"))

        return names, distance

    def train(self, data_root):
        batch_size = 128
        data = dataset.AddDataset(data_root, not self.feat_extract)
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             num_workers=12, pin_memory=True)

        num_elements = data.__len__()

        x = np.zeros((data.__len__(), self.model.num_features), dtype=np.float32)

        with torch.no_grad():
            for i, (images, filenames) in enumerate(loader):
                images = images.view(-1, 3, 224, 224).to(device=next(self.model.parameters()).device)

                out = self.model(images).cpu()

                x[i * batch_size : (i+1) * batch_size, :] = out.numpy()

        self.quantizer = faiss.IndexFlatL2(self.model.num_features)
        self.index = faiss.IndexIVFFlat(self.quantizer, self.model.num_features,
                                        int(np.sqrt(num_elements)))
        self.index.train(x)
        self.r.flushdb()

    def retrain(self):
        pass

    def save(self):
        faiss.write_index(self.index, self.name)

if __name__ == "__main__":
    model = densenet.Model(num_features=128)
    database = Database("db", model)
    database.add_dataset("image_folder/test/38/")

    # image = Image.open("image_folder/test/38/1884_1719624.png").convert('RGB')
    #
    # print(database.search(image))
