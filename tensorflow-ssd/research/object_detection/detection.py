import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import scipy.misc
import subprocess as sp

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images'
TEST_IMAGE_PATHS = sorted(list(os.listdir(PATH_TO_TEST_IMAGES_DIR)))
TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR + '/' +
                    p.replace('._', '') for p in TEST_IMAGE_PATHS]

video_path = "00001.mp4"
model_dir = "fine_tuned_model/saved_model"

def load_model():
  model = tf.saved_model.load(str(model_dir),None)
  model = model.signatures['serving_default']

  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  return output_dict

def show_inference(model, frame):
    # Actual detection.
    output_dict = run_inference_for_single_image(model, frame)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True)

    return frame

def generate_frames(model):
    # Load video file
    vidcap = cv2.VideoCapture(video_path)
    #pipeline_out = "appsrc ! videoconvert ! x264enc ! mpegtsmux ! queue ! hlssink target-d$
    pipeline_out = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=2 byte-stream=true ! flvmux ! rtmpsink location='rtmp://0.0.0.0:1935/show/stream"
    fourcc = cv2.VideoWriter_fourcc(*'H264') # For HLS
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, fourcc, fps, (width,height))
    success = True
#    command = ['ffmpeg',
#      '-y',
#      '-f', 'rawvideo',
#      '-vcodec','rawvideo',
#      '-pix_fmt', 'bgr24',
#      '-s', str(width)+'x'+str(height),
#      '-i', '-',
#      '-c:v', 'libx264',
#      '-pix_fmt', 'yuv420p',
#      '-preset', 'ultrafast',
#      '-f', 'flv',
#      'rtmp://localhost/show/stream']
#    proc = sp.Popen(command, stdin=sp.PIPE,shell=False)

    while success:
      #If a new model is present, excahnge this with the old model.
      if "new_model.pb" in os.listdir(model_dir):
            sp.call("mv " + model_dir + "/new_model.pb " + model_dir + "/saved_model.pb", shell=True)
            model = load_model()

      success, frame = vidcap.read()
      annotated_frame = show_inference(model, np.array(frame))
      writer.write(annotated_frame)
      #proc.stdin.write(frame.tostring())

if __name__ == '__main__':
    detection_model = load_model()
    generate_frames(detection_model)
