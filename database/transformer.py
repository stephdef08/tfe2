from transformers import ViTForImageClassification
from transformers import DeiTForImageClassification
import torch
from argparse import ArgumentParser, ArgumentTypeError
import database.dataset as dataset
from database.loss import MarginLoss
import time
import numpy as np
from torch import nn

class Model(torch.nn.Module):
    def __init__(self, num_features=128, batch_size=32, name='weights', eval=True, device='cuda:0'):
        super(Model, self).__init__()
        # self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224')

        # for param in self.model.parameters():
        #     param.requires_grad = False

        # for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
        #     module.eval()
        #     module.train = lambda _: None

        self.model.classifier = torch.nn.Linear(768, num_features)
        self.norm = torch.nn.functional.normalize

        self.model.to(device=device)

        if eval:
            self.model.eval()
            self.load_state_dict(torch.load(name))
        else:
            self.model.train()

        self.eval = eval
        self.device = device
        self.num_features = num_features
        self.name = name
        self.batch_size = batch_size

    def forward(self, input):
        return self.norm(self.model(input).logits, 1)

    def train_epochs(self, dir, epochs, sched, alan, reguliser):
        lr = 0.0001
        decay = 0.0004
        beta_lr = 0.0005
        gamma = 0.3

        data = dataset.DRDataset(dir, 2, alan, True)
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

                for i, (labels, encoding) in enumerate(loader):
                    images_gpu = encoding.to(device=self.device)
                    labels = labels.to(device=self.device)

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

                torch.save(self.state_dict(), self.name + '_' + str(epoch))

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
        '--file_name',
        default='weights'
    )

    parser.add_argument(
        '--training_data',
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

    m = Model(num_features=args.num_features, batch_size=args.batch_size,
              name=args.file_name, eval=False, device=device)

    m.train_epochs(args.training_data, args.num_epochs, args.scheduler, args.alan, args.reguliser)
