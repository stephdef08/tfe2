from flask import Flask, request, Response
from PIL import Image
from io import BytesIO
from base64 import encodebytes
import argparse
import json
from database.db import Database
from database.densenet import Model
from multiprocessing import Pool

app = Flask(__name__)

class Server:
    def __init__(self, ip, port, master_ip, master_port, model, num_features,
                 weights, use_dr, device, attention):
        self.ip = ip
        self.port = port
        self.master_ip = master_ip
        self.master_port = master_port
        self.model = Model(model, num_features=num_features, name=weights,
                           use_dr=use_dr, device=device, attention=attention)
        self.database = Database("db", self.model, load=True)

@app.route('/nearest_neighbours', methods=['POST'])
def nearest_neighbours():
    req = request.args
    names[request.environ['REMOTE_ADDR']], distance = \
        server.database.search(Image.open(request.files['file']),
                               int(req['nrt_neigh']))

    distance[0].tolist()
    return json.dumps({'distances': distance[0].tolist()})

@app.route('/retrieve_images')
def retrieve_labels():
    req = request.args

    images = []
    sizes = []
    for i in req['labels']:
        img = Image.open(names[request.environ['REMOTE_ADDR']][int(i)])
        bytes_io = BytesIO()
        img.save(bytes_io, 'png')
        bytes_io.seek(0)

        images.append(encodebytes(bytes_io.getvalue()).decode('ascii'))
        sizes.append(img.size)

    return json.dumps({'images': images, 'sizes': sizes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--master_ip',
        default='127.0.0.1'
    )

    parser.add_argument(
        '--master_port',
        default=8000
    )

    parser.add_argument(
        '--ip',
        default='127.0.0.1'
    )

    parser.add_argument(
        '--port',
        default=8001
    )

    parser.add_argument(
        '--model',
        default='densenet'
    )

    parser.add_argument(
        '--num_features',
        default=128,
        type=int
    )

    parser.add_argument(
        '--weights',
        default='weights/weights'
    )

    parser.add_argument(
        '--use_dr',
        action='store_true'
    )

    parser.add_argument(
        '--gpu_id',
        default=-1,
        type=int
    )

    parser.add_argument(
        '--attention',
        action='store_true'
    )

    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = 'cuda:' + str(args.gpu_id)
    else:
        device = 'cpu'

    server = Server(args.ip, args.port, args.master_ip, args.master_port,
                    args.model, args.num_features, args.weights, args.use_dr,
                    device, args.attention)

    names = {}

    app.run(host=args.ip, port=args.port)
