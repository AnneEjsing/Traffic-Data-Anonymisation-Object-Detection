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
import threading

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
PATH_TO_FACE_LABELS = "annotations/label_map_face.pbtxt"
category_index_face = label_map_util.create_category_index_from_labelmap(
    PATH_TO_FACE_LABELS, use_display_name=True)

PATH_TO_LICENSE_PLATE_LABELS = "annotations/label_map_license_plate.pbtxt"
category_index_license_plate = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LICENSE_PLATE_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'images'
TEST_IMAGE_PATHS = sorted(list(os.listdir(PATH_TO_TEST_IMAGES_DIR)))
TEST_IMAGE_PATHS = [PATH_TO_TEST_IMAGES_DIR + '/' +
                    p.replace('._', '') for p in TEST_IMAGE_PATHS]

video_path = "00001.mp4"
license_plate_model_dir = "fine_tuned_model/license_plate/saved_model"
face_model_dir = "fine_tuned_model/face/saved_model"

stream = ""
stop = None


def load_model():
    license_plate_model = tf.saved_model.load(str(license_plate_model_dir), None)
    license_plate_model = license_plate_model.signatures['serving_default']

    face_model = tf.saved_model.load(str(face_model_dir), None)
    face_model = face_model.signatures['serving_default']

    return license_plate_model, face_model

def prettify_output_dict(output_dict):
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)
    return output_dict

def run_inference_for_single_image(license_plate_model,face_model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    license_plate_output_dict = license_plate_model(input_tensor)
    face_output_dict = face_model(input_tensor)

    return prettify_output_dict(license_plate_output_dict), prettify_output_dict(face_output_dict)

def show_bounding_boxes(frame, output_dict, index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        index,
        use_normalized_coordinates=True)


def show_inference(license_plate_model, face_model, frame):
    # Actual detection.
    license_plate_output_dict, face_output_dict = run_inference_for_single_image(license_plate_model,face_model, frame)
    
    # Visualization of the results of a detection.
    #show_bounding_boxes(frame,license_plate_output_dict, category_index_license_plate)
    #show_bounding_boxes(frame,face_output_dict, category_index_face)

    # Uncomment this, if you want to see the detections on the image.
    #cv2.imshow('image', frame)
    #cv2.waitKey(1)
    frame = blur_frame(frame, license_plate_output_dict)
    frame = blur_frame(frame, face_output_dict)

    return frame


def blur_frame(frame, output_dict):
    im_width = len(frame[0])
    im_height = len(frame)

    for box, score in zip(output_dict['detection_boxes'], output_dict['detection_scores']):
        if score < 0.5:
            break

        (ymin, xmin, ymax, xmax) = box
        ymin, xmin, ymax, xmax = int(float(ymin) * im_height), int(float(xmin) * im_width), int(float(ymax) * im_height), int(float(xmax) * im_width)
        if (ymax - ymin <= 0) or (xmax - xmin <= 0):
            continue

        region = frame[ymin:ymax, xmin:xmax]
        blurred_region = cv2.blur(region, (21,21))
        frame[ymin:ymin + blurred_region.shape[0], xmin:xmin + blurred_region.shape[1]] = blurred_region

    return frame


def generate_frames(license_plate_model, face_model):
    # Load video file
    vidcap = cv2.VideoCapture(video_path)
    fps = 10
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    success = True
    command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', str(width)+'x'+str(height),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        stream ]
    proc = sp.Popen(command, stdin=sp.PIPE,shell=False)

    while success and not stop.is_set():
        # If a new model is present, excahnge this with the old model.
        if "new_face.pb" in os.listdir(face_model_dir):
            sp.call(f"mv {face_model_dir}/new_face_model.pb {face_model_dir}/saved_model.pb", shell=True)
            model = load_model()
        if "new_license_plate.pb" in os.listdir(license_plate_model_dir):
            sp.call(f"mv {license_plate_model_dir}/new_license_plate_model.pb {license_plate_model_dir}/saved_model.pb", shell=True)
            model = load_model()

        success, frame = vidcap.read()
        annotated_frame = show_inference(
            license_plate_model, face_model, np.array(frame))
        proc.stdin.write(annotated_frame.tostring())

    if stop.is_set():
        print("Stopping detection thread")
        
    proc.kill()


def start_detection(stream_endpoint, stop_event):
    global stream, stop
    stream = stream_endpoint
    stop = stop_event
    license_plate_model,face_model = load_model()
    generate_frames(license_plate_model, face_model)

if __name__ == '__main__':
    start_detection("rtmp://localhost:1935/show/stream", threading.Event()) #dummy threading event for when running locally
