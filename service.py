from aiohttp import web
import asyncio

routes = web.RouteTableDef()

@routes.post("/model/upload")
async def receive_model(request):
    data = await request.post()
    input_file = data['file']
    type = data['type']
    path = ""

    extension = input_file.filename.split('.')[-1]
    if extension == "pb":
        if type == "face":
            path = "tensorflow-ssd/fine_tuned_model/face/saved_model/new_face_model.pb"
        elif type == "license_plate":
            path = "tensorflow-ssd/fine_tuned_model/license/saved_model/new_license_plate_model.pb"
    elif extension == "m5":
        path = "keras-retinanet/new_model.h5"
    else: 
        return web.Response(status=415) #Unsupported media type


    with open(path, 'w+b') as f:
        content = input_file.file.read()
        f.write(content)

    return web.Response(status=200)

if __name__ == "__main__":  
    app = web.Application(client_max_size=0)
    app.add_routes(routes)
    web.run_app(app, host='0.0.0.0', port=5000)
