import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import database.dataset as dataset
import numpy as np
import time
from database.loss import MarginLoss

import kornia

from argparse import ArgumentParser, ArgumentTypeError

class Model(nn.Module):
    def __init__(self, model='densenet', eval=True, batch_size=32, num_features=128,
                 name='weights', use_dr=True, device='cuda:0', attention=False):
        super(Model, self).__init__()
        if model == 'densenet':
            self.conv_net = models.densenet121(pretrained=True).to(device=device)
        elif model == 'resnet':
            self.conv_net = models.resnet50(pretrained=True).to(device=device)

        for param in self.conv_net.parameters():
            param.requires_grad = False

        out_features = 4096 if use_dr else num_features
        if model == 'densenet':
            self.conv_net.classifier = nn.Linear(1024, out_features).to(device=device)
        elif model == 'resnet':
            self.conv_net.fc = nn.Linear(2048, out_features).to(device=device)

        self.relu = nn.LeakyReLU().to(device=device)

        if attention:
            self.att_layer = torch.nn.MultiheadAttention(out_features, 1).to(device=device)

        if use_dr:
            self.first_conv1 = nn.Conv2d(3, 96, kernel_size=8, padding=1, stride=16).to(device=device)
            self.first_conv2 = nn.MaxPool2d(3, 4, 1).to(device=device)

            self.second_conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=4, stride=32).to(device=device)
            self.second_conv2 = nn.MaxPool2d(7, 2, 3).to(device=device)

            self.linear = nn.Linear(7168, num_features).to(device=device)

        self.num_features = num_features
        self.use_dr = use_dr
        self.attention = attention
        self.norm = nn.functional.normalize

        self.name = name
        self.device = device

        if eval == True:
            self.load_state_dict(torch.load(name))
            self.eval()
            self.eval = True
        else:
            self.train()
            self.eval = False
            self.batch_size = batch_size


    def forward(self, input):
        tensor1 = self.conv_net(input)

        if self.attention:
            tensor1 = tensor1.unsqueeze(0)
            tensor1 = self.att_layer(tensor1, tensor1, tensor1)[0].view((-1, self.num_features))

        tensor1 = self.norm(self.relu(tensor1))

        if self.use_dr:
            tensor2 = self.first_conv1(input)
            tensor2 = self.first_conv2(tensor2)
            tensor2 = torch.flatten(tensor2, start_dim=1)

            tensor3 = self.second_conv1(input)
            tensor3 = self.second_conv2(tensor3)
            tensor3 = torch.flatten(tensor3, start_dim=1)

            tensor4 = self.norm(torch.cat((tensor2, tensor3), 1))

            return self.norm(self.linear(torch.cat((tensor1, tensor4), 1)))

        return tensor1

    def train_epochs(self, dir, epochs, sched, alan, reguliser):
        lr = 0.0001
        decay = 0.0004
        beta_lr = 0.0005
        gamma = 0.3

        data = dataset.DRDataset(dir, 2, alan)
        print(data.__len__())

        loss_function = MarginLoss(n_classes=len(data.classes), reguliser=reguliser)

        to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay},
                    {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

        optimizer = torch.optim.Adam(to_optim)

        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                            gamma=gamma)

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=4,
                                             pin_memory=True)

        loss_list = []
        try:
            for epoch in range(epochs):
                start_time = time.time()

                for i, (labels, images) in enumerate(loader):
                    images_gpu = images.to(device=self.device)
                    labels = labels.to(device=self.device)

                    out = self.forward(images_gpu)

                    loss = loss_function(out, labels)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())

                    if i % 100 == 0:
                        print("epoch {}, batch {}, loss = {}".format(epoch, i,
                                                                     np.mean(loss_list)))
                        loss_list.clear()

                print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_list),
                                                            time.time() - start_time))

                print("\n----------------------------------------------------------------\n")
                if sched != None:
                    scheduler.step()

                torch.save(self.state_dict(), self.name)

        except KeyboardInterrupt:
            print("Interrupted")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        type=int,
        help='number of features to use',
        default=128
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--model',
        help='feature extractor to use',
        default='densenet'
    )

    parser.add_argument(
        '--file_name',
        default='weights'
    )

    parser.add_argument(
        '--training_data',
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5
    )

    parser.add_argument(
        '--scheduler',
        default=None
    )

    parser.add_argument(
        '--attention',
        action='store_true'
    )

    parser.add_argument(
        '--alan',
        action='store_true'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--reguliser',
        default='contrastive_p'
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    m = Model(model=args.model, eval=False, batch_size=args.batch_size,
              num_features=args.num_features, name=args.file_name,
              use_dr=args.dr_model, attention=args.attention, device=device)

    m.train_epochs(args.training_data, args.num_epochs, args.scheduler, args.alan, args.reguliser)
