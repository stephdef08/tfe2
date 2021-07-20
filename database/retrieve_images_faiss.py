from database.db import Database
from argparse import ArgumentParser, ArgumentTypeError
import database.densenet as densenet
from PIL import Image
import time
import torch
import database.transformer as transformer

class ImageRetriever:
    def __init__(self, db_name, model):
        self.db = Database(db_name, model, True)

    def retrieve(self, image):
        return self.db.search(image)[0]

if __name__ == "__main__":
    usage = "python3 add_images.py --path <image_name> [--extractor <algorithm> --db_name <name> --num_features <num>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--db_name',
        help='name of the database',
        default='db'
    )

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=128,
        type=int
    )

    parser.add_argument(
        '--file_name',
        help='file that contains the weights of the network',
        default='weights'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--attention',
        action='store_true'
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print(usage)
        exit(-1)

    model = None

    if args.extractor != "transformer":
        model = densenet.Model(num_features=args.num_features, name=args.file_name,
                               use_dr=args.dr_model, attention=args.attention, device=device)
    else:
        model = transformer.Model(num_features=args.num_features, name=args.file_name, device=device)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    retriever = ImageRetriever(args.db_name, model)

    names = retriever.retrieve(Image.open(args.path).convert('RGB'))
    for n in names:
        Image.open(n).show()
    print(names)
