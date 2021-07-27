from fastapi import FastAPI
import uvicorn
import argparse
import json
import multiprocessing

app = FastAPI()

class Master:
    def __init__(self, ip, port, servers):
        self.ip = ip
        self.port = port
        self.servers = set(servers)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--ip',
    default='127.0.0.1'
)

parser.add_argument(
    '--port',
    default=8000,
    type=int
)

parser.add_argument(
    '--server_addresses',
    nargs='+'
)

args = parser.parse_args()


if __name__ != '__main__':
    master = Master(args.ip, args.port, args.server_addresses)

@app.get('/servers_list')
def servers_list():
    return {'list': list(master.servers)}

if __name__ == '__main__':
    uvicorn.run('master:app', host=args.ip, port=args.port, reload=True,
                debug=True, workers=multiprocessing.cpu_count())
