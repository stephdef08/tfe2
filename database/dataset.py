from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import copy
from collections import defaultdict
import kornia
from transformers import ViTFeatureExtractor
from transformers import DeiTFeatureExtractor
# import cv2

class DRDataset(Dataset):
    def __init__(self, root, samples_per_class, alan=False, transformer=False):
        if alan == False:
            self.classes = os.listdir(root)
        else:
            self.classes = []
            for c in os.listdir(root):
                for cls in os.listdir(os.path.join(root, c, 'train')):
                    self.classes.append(os.path.join(root, c, 'train', cls))
        # self.classes = self.classes[:len(self.classes) // 2 + 1]
        self.conversion = {x: i for i, x in enumerate(self.classes)}
        self.conv_inv = {i: x for i, x in enumerate(self.classes)}
        self.image_dict = {}
        self.image_list = defaultdict(list)

        print("================================")
        print("Loading dataset")
        print("================================")

        i = 0
        if alan == False:
            for c in self.classes:
                for dir, subdirs, files in os.walk(root + c):
                    # if "camelyon16_0" in dir:
                    #     continue
                    for file in files:
                        # img = Image.open(os.path.join(dir, file)).convert('RGB')
                        # img = transforms.Resize((224, 224))(img)
                        img = os.path.join(dir, file)
                        cls = dir[dir.rfind("/") + 1:]
                        # if "camelyon" in cls:
                        #     gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                        #     var = np.var(gray)
                        #     if var >= 2500 and var <= 4000:
                        #         self.image_dict[i] = (img, self.conversion[cls])
                        #         self.image_list[self.conversion[cls]].append(img)
                        # else:
                        self.image_dict[i] = (img, self.conversion[cls])
                        self.image_list[self.conversion[cls]].append(img)
                        i += 1
        else:
            for c in self.classes:
                for dir, subdirs, files in os.walk(c):
                    for file in files:
                        img = os.path.join(dir, file)
                        self.image_dict[i] = (img, self.conversion[dir])
                        self.image_list[self.conversion[dir]].append(img)
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
            # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224',
            #                                                              image_mean=[0.485, 0.456, 0.406],
            #                                                              image_std=[0.229, 0.224, 0.225])
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
        # self.transform = kornia.augmentation.AugmentationSequential(
        #     kornia.augmentation.RandomVerticalFlip(p=.5),
        #     kornia.augmentation.RandomHorizontalFlip(p=.5),
        #     kornia.augmentation.ColorJitter(brightness=0, contrast=0, saturation=.2, hue=.1),
        #     kornia.augmentation.RandomResizedCrop((224, 224), scale=(.7,1)),
        #     kornia.augmentation.RandomElasticTransform(alpha=(1.5, 1.5)),
        #     kornia.augmentation.Normalize(
        #         mean=torch.Tensor([0.485, 0.456, 0.406]),
        #         std=torch.Tensor([0.229, 0.224, 0.225])
        #     ),
        #     return_transform=False,
        #     same_on_batch=False
        # )

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

        # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224',
        #                                                              image_mean=[0.485, 0.456, 0.406],
        #                                                              image_std=[0.229, 0.224, 0.225])
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
