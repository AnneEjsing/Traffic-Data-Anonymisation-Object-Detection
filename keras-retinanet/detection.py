import cv2
import numpy as np
import time
import subprocess as sp
import os

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

model_path = 'model.h5'
video_path = "00001.mp4"
labels_to_names = {0: 'head', 1: 'license_plate'}
blur = True

def detect_one_image(image, model):
  draw = image.copy()

  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image,540,960)

  # process image
  start = time.time()
  boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
  #print("processing time: ", time.time() - start)

  # correct for image scale
  boxes /= scale

  # visualize detections
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
      if score < 0.5:
          break
      
      (x1,y1,x2,y2) = box
      x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
      if blur:
          #Extracting the area to blur
          region = draw[y1:y2,x1:x2]

          #Create a blurred image of the area
          #   OBS: Region size has to be odd numbers!!!
          blurred_region = cv2.blur(region,(21,21))

          #Set the area of interest to the blurred image of that area.
          draw[y1:y1+blurred_region.shape[0],x1:x1+blurred_region.shape[1]] = blurred_region

      else:
          color = label_color(label)
          b = box.astype(int)
          draw_box(draw, b, color=color)
          caption = "{} {:.3f}".format(labels_to_names[label], score)
          draw_caption(draw, b, caption)
  return draw

def generate_frames(model):
  # Load video file
  vidcap = cv2.VideoCapture(video_path)
  pipeline_out = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=2 byte-stream=true ! flvmux ! rtmpsink location='rtmp://0.0.0.0:1935/show/stream" # Create RTMP 
  fourcc = cv2.VideoWriter_fourcc(*'H264') # For HLS
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  writer = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, fourcc, fps, (width,height))
  success = True

  while success:
    # If a new model is present, excahnge this with the old model.
    if "new_model.h5" in os.listdir():
        sp.call("mv new_model.m5 model.m5", shell=True)
        model = models.load_model(model_path, backbone_name="resnet50")
    success, frame = vidcap.read()
    annotated_frame = detect_one_image(frame, model)
    writer.write(frame)

if __name__ == '__main__':
    model = models.load_model(model_path, backbone_name="resnet50") # https://github.com/fizyr/keras-retinanet/releases
    generate_frames(model)


