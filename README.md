# tfe2
parts of codes are taken from:
- https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
- https://github.com/euwern/proxynca_pp
- https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py

## Requirements

The library was tested with python 3.9
```
pip install -r requirements.txt
```

Two more external libraries have to be installed:
- Cytomine (https://doc.cytomine.org/dev-guide/clients/python/installation)
- OpenSlide (https://pypi.org/project/openslide-python/)

The required libraries are in the file requirements.txt

## Training a new model
To train a new model launch the script `database/models.py`

```bash
python database/models.py [arguments]
```

The following arguments can be given in the command line:
- --num_features (default: 128): the size of the last linear layer (i.e. the number of features)
- --batch_size (default: 32): the batch size used for training
- --model (default: densenet): densenet, resnet or transformer
- --weights (default: weights): the file that will contain the weights, a different file is saved for every epoch, with the appended number of the epoch
- --training_data (required): where the training images are stored
- --dr_model (flag): use the two paralel shallow convolutional networks (not for the visual transformer)
- --num_epochs (default: 5): number of epochs
- --scheduler (default: None): exponential, step
- --gpu_id (default: 0): the id of the gpu to use for training
- --loss (default: margin): margin, proxy_nca_pp, softmax or deep_ranking
- --freeze (flag): freeze the weights of the model during training (not for the last layer and the shallow convolutional networks)
- --generalise (flag): train on only half the classes
- --lr, --decay, --beta_lr, --gamma, --lr_proxies: parameters related to the training

The folder that contains the training images should be organised as follows:
```
folder:
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

## Indexing images to the database
```bash
redis-server
python database/add_images [arguments]
```

The following arguments can be given in the command line:
- --path (required): path to the images to add
- --extractor (default: densenet): densenet, resnet or transformer
- --weights (default: weights): file storing the weights of the network
- --db_nam (default: db): will regroup different files needed for the database under a same name. Name of the database (e.g. storage/database)
- --num_features (default: 128): the size of the last linear layer (i.e. the number of features)
- --rewrite (flag): erase the previous content of the database, otherwise, add to the existing data
- --dr_model (flag): use the two paralel shallow convolutional networks (not for the visual transformer)
- --gpu_id (default: 0): the id of the gpu on which the extractor will be loaded

## Retrieve images
The redis server that was used to index the images must be running
```bash
python database/retrieve_images [arguments]
```

The following arguments can be given in the command line:
- --path (required): path to the query image
- --extractor (default: densenet): densenet, resnet or transformer
- --db_name (default: db): name of the database
- --weights (default: weights): file storing the weights of the extractor
- --dr_model (flag): whether to use the two shallow convolutional networks or not
- --gpu_id (default: 0): id of the gpu on which the extractor is loaded
- --nrt_neigh (default: 10): number of images to retrieve

The folder that contains the images must have the same structure that the one used for training

## Testing the accuracy
The redis server that was used to index the images must be running
```bash
python database/test_accuracy.py [arguments]
```

The following arguments can be given in the command line:
- --num_features (default: 128): the size of the last linear layer (i.e. the number of features)
- --path (required): path to the query images
- --extractor (default: densenet): densenet, resnet or transformer
- --dr_model (flag): use the two paralel shallow convolutional networks (not for the visual transformer)
- --weights (default: weights): file storing the weights of the network
- --db_name (default: db): name of the database
- --gpu_id (default: 0): id of the gpu on which the extractor is loaded
- --measure (default: random): random, remove or all. `Random` stands for the random sampling of query images in each classes, `remove` stands for the removal of camelyon16_0 and janowczyk6_0, and for `all`, all the images are considered
- --generalise (flag): use only half the classes to compute the accuracy

The folder that contains the images must have the same structure that the one used for training

## Using the REST API
Images can already be indexed in the database before launching a server

The master is launched be executing the following command:
```bash
python rest/master.py [arguments]
```

The following arguments can be given in the command line:
- --ip (default: 127.0.0.1): exposed ip address of the master
- --port (default: 8000): port used for the communication with the clients
- --http (flag): use the HTTP protocol instead of HTTPS
- --host (required): Cytomine host

Afterwards, each image server can be launched independently, by first launching the redis server associated to the image server and executing the following command
```bash
python rest/server.py [arguments]
```
- --master_ip (default: 127.0.0.1): ip of the master server
- --master_port (default: 8000): port of the master
- --ip (default: 127.0.0.1): exposed ip address of the server
- --port (default: 8001): port used for the communication with the master
- --extractor (default: densenet): densenet, resnet or transformer
- --num_features (default: 128): the size of the last linear layer (i.e. the number of features)
- --weights (default: weights): file storing the weights of the network
- --use_dr (flag): use the two paralel shallow convolutional networks (not for the visual transformer)
- --gpu_id (default: 0): id of the gpu on which the extractor is loaded
- --folder (default: images): folder where the images will be stored
- --http (flag): use the HTTP protocol instead of HTTPS
- --db_name (default: db): name of the database
- --server_name (required): name of the server. Two different servers should have a different name

The REST API can be tested with FastAPI documentation, by going at the address `[http|https]://master_ip:master_port/docs`

## Training a Faiss index
If the server is running, it must first be shut down

The following command is executed to train the index on the whole dataset indexed locally

```bash
python database/db.py [arguments]
```

with the following arguments:
- --extractor (default: densenet): densenet, resnet or transformer
- --weights (required): name of the file containing the weights of the model
- --db_name (required): name of the database
- --unlabeled (flag): if raised, train the unlabeled index instead of the labeled
