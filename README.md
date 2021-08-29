# tfe2
parts of codes are taken from:
- https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
- https://github.com/euwern/proxynca_pp
- https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py

The required libraries are in the file requirements.txt

## Training a new model
To train a new model launch the script `database/models.py`

```bash
python database/models.py [arguments]
```

The following arguments can be given in the command line:
- --num_features: the size of the last linear layer (i.e. the number of features)
-  --batch_size: the batch size used for training
-  --model: densenet, resnet or transformer
-  --weights: the file that will contain the weights, a different file is saved for every epoch, with the appended number of the epoch
-  --training_data: where the training images are stored
-  --dr_model: use the two paralel shallow convolutional networks (not for the visual transformer)
-  --num_epochs: number of epochs
-  --scheduler: exponential, step
-  --gpu_id: the id of the gpu to use for training
-  --loss: margin, proxy_nca_pp, softmax or deep_ranking
-  --freeze: freeze the weights of the model during training (not for the last layer and the shallow convolutional networks)
-  --generalise: train on only half the classes
-  --lr, --decay, --beta_lr, --gamma, --lr_proxies: parameters related to the training

The folder that contains the training images should be organised as follows:
```
folder
|------ class1:
          |------ image1
          |------ image2
          |------ ...
|------ class2:
          |------ image1
          |------ image2
          |------ ..
|------ ...
```

## Adding images to the database
```bash
redis-server
python database/add_images [arguments]
```

The following arguments can be given in the command line:
- --path: 
