from database.db import Database
from argparse import ArgumentParser, ArgumentTypeError
import database.densenet as densenet
import torch
import database.transformer as transformer
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
        '--file_name',
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
        '--attention',
        action='store_true'
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

    model = None

    if args.extractor != 'transformer':
        model = densenet.Model(model=args.extractor, use_dr=args.dr_model, num_features=args.num_features, name=args.file_name,
                               device=device, attention=args.attention)
    else:
        model = transformer.Model(num_features=args.num_features, name=args.file_name, device=device)

    if model is None:
        print("Unkown feature extractor")
        exit(-1)

    database = Database(args.db_name, model, load= not args.rewrite, transformer=args.extractor=='transformer')

    name_list = ['janowczyk2_0', 'janowczyk2_1', 'lbpstroma_0', 'lbpstroma_1', 'patterns_no_aug_0', 'patterns_no_aug_1',
                 'mitos2014_0', 'mitos2014_1', 'mitos2014_2', 'ulg_lbtd_lba0', 'ulg_lbtd_lba1', 'ulg_lbtd_lba2', 'ulg_lbtd_lba3',
                 'ulg_lbtd_lba4', 'ulg_lbtd_lba5', 'ulg_lbtd_lba6', 'ulg_lbtd_lba7', 'iciar18_micro0', 'iciar18_micro1',
                 'iciar18_micro2', 'iciar18_micro3', 'tupac_mitosis0', 'tupac_mitosis1', 'camelyon16_0', 'camelyon16_1',
                 'umcm_colorectal_01', 'umcm_colorectal_02', 'umcm_colorectal_03', 'umcm_colorectal_04', 'umcm_colorectal_05',
                 'umcm_colorectal_06', 'umcm_colorectal_07', 'umcm_colorectal_08', 'warwick_crc0']

    ls = os.listdir(os.path.join(args.path))
    for n in name_list:
        ls.remove(n)
    import os
    names_list = []
    for n in ls:
        names_list +=  [os.path.join(n, img) for img in os.listdir(os.path.join(args.path, n))]

    database.add_dataset(args.path)
