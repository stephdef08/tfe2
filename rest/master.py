from flask import Flask, request, Response
import argparse
import json

app = Flask(__name__)

class Master:
    def __init__(self, ip, port, servers):
        self.ip = ip
        self.port = port
        self.servers = set(servers)


@app.route('/')
def general():
    req = request.args

    if 'servers_list' in req:
        return json.dumps({'list': list(master.servers)})

    return Response(status=404)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ip',
        default='127.0.0.1'
    )

    parser.add_argument(
        '--port',
        default=8000
    )

    parser.add_argument(
        '--server_addresses',
        nargs='+'
    )

    args = parser.parse_args()

    master = Master(args.ip, args.port, args.server_addresses)

    app.run(host=args.ip, port=args.port)
