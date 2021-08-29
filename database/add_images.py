from database.db import Database
from argparse import ArgumentParser, ArgumentTypeError
import database.models as models
import torch
import time
import os

if __name__ == "__main__":
    usage = "python3 add_images.py --path <folder> [--extractor <algorithm> --db_name <name> --num_features <num>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the folder that contains the images to add',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
        default='densenet'
    )

    parser.add_argument(
        '--weights',
        help='file that contains the weights of the network',
        default='weights'
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
        '--rewrite',
        help='if the database already exists, rewrite it',
        action='store_true'
    )

    parser.add_argument(
        '--dr_model',
        action="store_true"
    )

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int
    )

    parser.add_argument('--labeled', action='store_true')

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    if args.path is None:
        print(usage)
        exit(-1)

    if not os.path.isdir(args.path):
        print("The path mentionned is not a folder")
        exit(-1)

    model = models.Model(model=args.extractor, use_dr=args.dr_model, num_features=args.num_features, name=args.weights,
                           device=device)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    database = Database(args.db_name, model, load= not args.rewrite, transformer=args.extractor=='transformer')

    database.add_dataset(args.path)
