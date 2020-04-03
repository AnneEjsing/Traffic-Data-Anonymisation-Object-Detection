from werkzeug.datastructures import FileStorage
import os
from aiohttp import web
import aiohttp_cors
import asyncio
import requests
import subprocess as sp

routes = web.RouteTableDef()

@routes.post("/model/upload")
async def receive_model(request):
    data = await request.post()
    path = "tensorflow-ssd/fine_tuned_model/saved_model/new_model.pb"

    input_file = data['file']

    with open(path, 'w') as f:
        f.write(input_file)

    return web.Response(status=200)

if __name__ == "__main__":  
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, host='0.0.0.0', port=5000)
