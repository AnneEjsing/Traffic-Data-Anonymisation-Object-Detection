from aiohttp import web
import asyncio
from tensorflow_ssd.detection import start_detection
import threading

routes = web.RouteTableDef()

detection_thread = None
stop_event = None

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
            path = "tensorflow-ssd/fine_tuned_model/license_plate/saved_model/new_license_plate_model.pb" 
    elif extension == "m5":
        path = "keras-retinanet/new_model.h5"
    else: 
        return web.Response(status=415) #Unsupported media type


    with open(path, 'w+b') as f:
        content = input_file.file.read()
        f.write(content)

    return web.Response(status=200)

@routes.post("/start")
async def start_stream(request):
    global detection_thread, stop_event
    data = await request.json()
    stream_endpoint = data['reciever']

    if detection_thread != None:
        stop_event.set()
        detection_thread.join()

    stop_event = threading.Event()
    detection_thread = threading.Thread(target=start_detection, args=(stream_endpoint, stop_event))
    detection_thread.start()

    return web.Response(status=200)


if __name__ == "__main__":
    app = web.Application(client_max_size=0)
    app.add_routes(routes)
    web.run_app(app, host='0.0.0.0', port=5000)
