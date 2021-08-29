from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import copy
from collections import defaultdict
from transformers import DeiTFeatureExtractor

# https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py

class DRDataset(Dataset):

    def __init__(self, root='image_folder', transform=None):
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                    transforms.RandomResizedCrop(224, scale=(.8,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.root = root
        self.transform = transform
        self.rev_dict = {}
        self.image_dict = {}
        self.big_dict = {}
        L = []

        self.num_classes = 0

        self.num_elements = 0

        for i, j in enumerate(os.listdir(os.path.join(root))):
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(root, j)))
            for k in os.listdir(os.path.join(root, j)):
                self.big_dict[self.num_elements] = (k, i)
                self.num_elements += 1

            self.num_classes += 1

    def _sample(self, idx):
        im, im_class = self.big_dict[idx]
        im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
        numbers = list(range(im_class)) + list(range(im_class+1, self.num_classes))
        class3 = np.random.choice(numbers)
        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
        p1 = os.path.join(self.root, self.rev_dict[im_class], im)
        p2 = os.path.join(self.root, self.rev_dict[im_class], im2)
        p3 = os.path.join(self.root, self.rev_dict[class3], im3)
        return [p1, p2, p3]

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        paths = self._sample(idx)
        images = []
        for i in paths:
            tmp = Image.open(i).convert('RGB')
            tmp = self.transform(tmp)
            images.append(tmp)

        return (images[0], images[1], images[2])

class TrainingDataset(Dataset):
    def __init__(self, root, samples_per_class, generalise, transformer=False):
        self.classes = os.listdir(root)
        if generalise:
            self.classes = self.classes[:len(self.classes) // 2 + 1]
        self.conversion = {x: i for i, x in enumerate(self.classes)}
        self.conv_inv = {i: x for i, x in enumerate(self.classes)}
        self.image_dict = {}
        self.image_list = defaultdict(list)

        print("================================")
        print("Loading dataset")
        print("================================")
        i = 0
        for c in self.classes:
            for dir, subdirs, files in os.walk(os.path.join(root, c)):
                for file in files:
                    img = os.path.join(dir, file)
                    cls = dir[dir.rfind("/") + 1:]
                    self.image_dict[i] = (img, self.conversion[cls])
                    self.image_list[self.conversion[cls]].append(img)
                    i += 1

        if not transformer:
            self.transform = transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(.5),
                        transforms.RandomHorizontalFlip(.5),
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                        transforms.RandomResizedCrop(224, scale=(.7,1)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ]
                )
        else:
            self.transform = transforms.Compose(
                    [
                        transforms.RandomVerticalFlip(.5),
                        transforms.RandomHorizontalFlip(.5),
                        transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                        transforms.RandomResizedCrop(224, scale=(.7,1)),
                        transforms.ToTensor()
                    ]
                )

            self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                          size=224, do_center_crop=False,
                                                                          image_mean=[0.485, 0.456, 0.406],
                                                                          image_std=[0.229, 0.224, 0.225])

        self.transformer = transformer

        self.samples_per_class = samples_per_class
        self.current_class = np.random.choice(self.classes)
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        self.is_init = True

    def __len__(self):
        return len(self.image_dict)

    # https://github.com/Confusezius/Deep-Metric-Learning-Baselines/blob/60772745e28bc90077831bb4c9f07a233e602797/datasets.py#L428
    def __getitem__(self, idx):
        if self.is_init:
            self.current_class = self.classes[idx % len(self.classes)]
            self.is_init = False

        if self.samples_per_class == 1:
            img = Image.open(self.image_dict[idx][0]).convert('RGB')
            return self.image_dict[idx][0], self.transform(img)

        if self.n_samples_drawn == self.samples_per_class:
            counter = [cls for cls in self.classes if cls not in self.classes_visited]

            self.current_class = counter[idx % len(counter)]
            self.classes_visited = self.classes_visited[1:] + [self.current_class]
            self.n_samples_drawn = 0

        class_nbr = self.conversion[self.current_class]

        class_sample_idx = idx % len(self.image_list[class_nbr])
        self.n_samples_drawn += 1

        img = Image.open(self.image_list[class_nbr][class_sample_idx]).convert('RGB')

        if self.transformer:
            img = self.transform(img)
            return class_nbr, self.feature_extractor(images=img, return_tensors='pt')['pixel_values']

        return class_nbr, self.transform(img)

class AddDataset(Dataset):
    def __init__(self, root, transformer=False):
        self.root = root
        self.list_img = []
        self.transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
                    transforms.RandomResizedCrop(224, scale=(.7,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.transformer = transformer

        self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                      size=224, do_center_crop=False,
                                                                      image_mean=[0.485, 0.456, 0.406],
                                                                      image_std=[0.229, 0.224, 0.225])

        self.classes = os.listdir(root)
        # self.classes = self.classes[:len(self.classes) // 2 + 1]

        # for cls in self.classes:
        for subdir, dirs, files in os.walk(root):
            for f in files:
                self.list_img.append(os.path.join(subdir, f))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        img = Image.open(self.list_img[idx]).convert('RGB')

        if not self.transformer:
            return self.transform(img), self.list_img[idx]

        return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], self.list_img[idx]

class AddDatasetList(Dataset):
    def __init__(self, id, name_list, server_name='', transformer=False):
        self.list_img = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.transformer = transformer

        self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224',
                                                                      size=224, do_center_crop=False,
                                                                      image_mean=[0.485, 0.456, 0.406],
                                                                      image_std=[0.229, 0.224, 0.225])
        self.server_name = server_name
        self.id = id

        for n in name_list:
            self.list_img.append(n[0])

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        if self.server_name == '':
            img = Image.open(os.path.join(self.list_img[idx], str(idx+self.id) + '.png')).convert('RGB')
            if not self.transformer:
                return self.transform(img), os.path.join(self.list_img[idx], str(idx+self.id)  + '.png')
            return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], os.path.join(
                self.list_img[idx], str(idx+self.id)  + '.png')

        img = Image.open(os.path.join(self.list_img[idx], self.server_name + '_' + str(idx+self.id) + '.png')).convert('RGB')
        if not self.transformer:
            return self.transform(img), os.path.join(self.list_img[idx],
                                                     self.server_name + '_' + str(idx+self.id)  + '.png')
        return self.feature_extractor(images=img, return_tensors='pt')['pixel_values'], os.path.join(
            self.list_img[idx], self.server_name + '_' + str(idx+self.id)  + '.png')

class AddSlide(Dataset):
    def __init__(self, patches, slide):
        self.patches = patches
        self.slide = slide
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    def __len__(self):
        if self.slide.level_count > 1:
            return self.patches.shape[0] * 2
        else:
            return self.patches.shape[0]

    def __getitem__(self, key):
        if key < self.patches.shape[0]:
            return self.transform(self.slide.read_region((self.patches[key, 1] * 224, self.patches[key, 0] * 224), 0,
                                                         (224, 224)).convert('RGB'))
        else:
            return self.transform(self.slide.read_region((self.patches[key-self.patches.shape[0], 1] * 224, self.patches[key-self.patches.shape[0], 0] * 224), 1,
                                                         (224, 224)).convert('RGB'))
