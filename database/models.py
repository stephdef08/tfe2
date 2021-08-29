import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from transformers import DeiTForImageClassification
import database.dataset as dataset
import numpy as np
import time
from database.loss import MarginLoss, ProxyNCA_prob, NormSoftmax

from argparse import ArgumentParser, ArgumentTypeError

class Model(nn.Module):
    def __init__(self, model='densenet', eval=True, batch_size=32, num_features=128,
                 name='weights', use_dr=True, device='cuda:0', freeze=False):
        super(Model, self).__init__()
        if model == 'densenet':
            self.forward_function = self.forward_conv
            self.conv_net = models.densenet121(pretrained=True).to(device=device)
        elif model == 'resnet':
            self.forward_function = self.forward_conv
            self.conv_net = models.resnet50(pretrained=True).to(device=device)
        elif model == 'transformer':
            self.forward_function = self.forward_transformer
            self.model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224').to(device=device)

        if freeze and model != 'transformer':
            for param in self.conv_net.parameters():
                param.requires_grad = False
        elif freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if use_dr and model != 'transformer':
            out_features = 4096
            if model == 'densenet':
                self.conv_net.classifier = nn.Linear(1024, out_features).to(device=device)
            elif model == 'resnet':
                self.conv_net.fc = nn.Linear(2048, out_features).to(device=device)

            self.first_conv1 = nn.Conv2d(3, 96, kernel_size=8, padding=1, stride=16).to(device=device)
            self.first_conv2 = nn.MaxPool2d(3, 4, 1).to(device=device)

            self.second_conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=4, stride=32).to(device=device)
            self.second_conv2 = nn.MaxPool2d(7, 2, 3).to(device=device)

            self.linear = nn.Linear(7168, num_features).to(device=device)
            self.use_dr = True
        else:
            if model == 'densenet':
                self.conv_net.classifier = nn.Linear(1024, num_features).to(device=device)
            elif model == 'resnet':
                self.conv_net.fc = nn.Linear(2048, num_features).to(device=device)
            elif model == 'transformer':
                self.model.classifier = torch.nn.Linear(768, num_features).to(device=device)
                for module in filter(lambda m: type(m) == nn.LayerNorm, self.model.modules()):
                    module.eval()
                    module.train = lambda _: None
            self.use_dr = False

        self.num_features = num_features
        self.norm = nn.functional.normalize

        self.transformer = model == 'transformer'

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

    def forward_conv(self, input):
        tensor1 = self.conv_net(input)

        tensor1 = self.norm(tensor1)

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

    def forward_transformer(self, input):
        return self.norm(self.model(input).logits, 1)

    def forward(self, input):
        return self.forward_function(input)

    def train_epochs(self, dir, epochs, sched, loss, generalise, lr, decay, beta_lr, gamma, lr_proxies):
        data = dataset.TrainingDataset(dir, 2, generalise, self.transformer)
        print('Size of dataset', data.__len__())

        if loss == 'margin':
            loss_function = MarginLoss(n_classes=len(data.classes))

            to_optim = [{'params':self.parameters(),'lr':lr,'weight_decay':decay},
                        {'params':loss_function.parameters(), 'lr':beta_lr, 'weight_decay':0}]

            optimizer = torch.optim.Adam(to_optim)
        elif loss == 'proxy_nca_pp':
            loss_function = ProxyNCA_prob(len(data.classes), self.num_features, 3, device)

            to_optim = [
                {'params':self.parameters(), 'weight_decay':0},
                {'params':loss_function.parameters(), 'lr': lr_proxies},
            ]

            optimizer = torch.optim.Adam(to_optim, lr=lr, eps=1)
        elif loss == 'softmax':
            loss_function = NormSoftmax(0.05, len(data.classes), self.num_features, lr_proxies, self.device)

            to_optim = [
                {'params':self.parameters(),'lr':lr,'weight_decay':decay},
                {'params':loss_function.parameters(),'lr':lr_proxies,'weight_decay':decay}
            ]

            optimizer = torch.optim.Adam(to_optim)

        if sched == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif sched == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2, epochs],
                                                            gamma=gamma)

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=12,
                                             pin_memory=True)

        loss_list = []
        try:
            for epoch in range(epochs):
                start_time = time.time()

                for i, (labels, images) in enumerate(loader):
                    images_gpu = images.to(device=self.device)
                    labels = labels.to(device=self.device)

                    if not self.transformer:
                        out = self.forward(images_gpu)
                    else:
                        out = self.forward(images_gpu.view(-1, 3, 224, 224))

                    loss = loss_function(out, labels)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())

                print("epoch {}, loss = {}, time {}".format(epoch, np.mean(loss_list),
                                                            time.time() - start_time))

                print("\n----------------------------------------------------------------\n")
                if sched != None:
                    scheduler.step()

                torch.save(self.state_dict(), self.name)

        except KeyboardInterrupt:
            print("Interrupted")

    def train_dr(self, data, num_epochs, lr):
        data = dataset.DRDataset(data)
        print('Size of dataset', data.__len__())

        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=12,
                                             pin_memory=True)
        loss_function = torch.nn.TripletMarginLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_list = []
        try:
            for epoch in range(num_epochs):
                start_time = time.time()

                for i, (image0, image1, image2) in enumerate(loader):
                    image0 = image0.to(device='cuda:0')
                    image1 = image1.to(device='cuda:0')
                    image2 = image2.to(device='cuda:0')

                    out0 = self.forward(image0).cpu()
                    out1 = self.forward(image1).cpu()
                    out2 = self.forward(image2).cpu()

                    loss = loss_function(out0, out1, out2)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    loss_list.append(loss.item())

                print("epoch {}, batch {}, loss = {}".format(epoch, i,
                                                             np.mean(loss_list)))
                loss_list.clear()
                print("time for epoch {}".format(time.time()- start_time))

                torch.save(self.state_dict(), self.name)

                if (epoch + 1) % 4:
                    lr /= 2
                    for param in optimizer.param_groups:
                        param['lr'] = lr
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
        '--weights',
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
        default=None,
        help='<exponential, step>'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument(
        '--loss',
        default='margin',
        help='<margin, proxy_nca_pp, softmax, deep_ranking>'
    )

    parser.add_argument(
        '--freeze',
        action='store_true'
    )

    parser.add_argument(
        '--generalise',
        action='store_true',
        help='train on only half the classes of images'
    )

    parser.add_argument(
        '--lr',
        default=0.0001,
        type=float
    )

    parser.add_argument(
        '--decay',
        default=0.0004,
        type=float
    )

    parser.add_argument(
        '--beta_lr',
        default=0.0005,
        type=float
    )

    parser.add_argument(
        '--gamma',
        default=0.3,
        type=float
    )

    parser.add_argument(
        '--lr_proxies',
        default=0.00001,
        type=float
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    m = Model(model=args.model, eval=False, batch_size=args.batch_size,
              num_features=args.num_features, name=args.weights,
              use_dr=args.dr_model, device=device, freeze=args.freeze)

    if args.loss == 'deep_ranking':
        m.train_dr(args.training_data, args.num_epochs, args.lr)
    else:
        m.train_epochs(args.training_data, args.num_epochs, args.scheduler, args.loss, args.generalise,
                       args.lr, args.decay, args.beta_lr, args.gamma, args.lr_proxies)
