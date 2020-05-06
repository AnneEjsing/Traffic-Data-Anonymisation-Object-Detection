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

face_model_dir = 'fine_tuned_model/face/'
license_plate_model_dir = 'fine_tuned_model/license_plate/'
video_path = "00001.mp4"
labels_to_names = {0: 'license_plate', 1: 'face'}
blur = True


def detect_one_image(image, face_model, license_plate_model):
    draw = image.copy()

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image, 540, 960)

    # process image
    start = time.time()
    face_boxes, face_scores, face_labels = face_model.predict_on_batch(
        np.expand_dims(image, axis=0))
    license_plate_boxes, license_plate_scores, license_plate_labels = license_plate_model.predict_on_batch(
        np.expand_dims(image, axis=0))

    # correct for image scale
    face_boxes /= scale
    license_plate_boxes /= scale
    draw = blur_detections(draw, face_boxes, face_scores, face_labels)
    draw = blur_detections(draw, license_plate_boxes,
                           license_plate_scores, license_plate_labels)

    # Uncomment this, if you want to see the detections on the image.
    #cv2.imshow('image', draw)
    #cv2.waitKey(1)

    return draw


def blur_detections(draw, boxes, scores, labels):
  # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
      # scores are sorted so we can break
        if score < 0.5:
            break
        (x1, y1, x2, y2) = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if blur:
            # Extracting the area to blur
            region = draw[y1:y2, x1:x2]

            # Create a blurred image of the area
            #   OBS: Region size has to be odd numbers!!!
            blurred_region = cv2.Blur(region, (21, 21))

            # Set the area of interest to the blurred image of that area.
            draw[y1:y1+blurred_region.shape[0], x1:x1 +
                blurred_region.shape[1]] = blurred_region

        else:
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
    return draw


def generate_frames(face_model, license_plate_model):
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
        'rtmp://localhost:1935/show/test']
    proc = sp.Popen(command, stdin=sp.PIPE, shell=False)

    while success:
        # If a new model is present, excahnge this with the old model.
                # If a new model is present, excahnge this with the old model.
        if "new_face_model.h5" in os.listdir(face_model_dir):
            sp.call(f"mv {face_model_dir}/new_face_model.h5 {face_model_dir}/model.h5", shell=True)
            face_model = models.load_model(face_model_dir+'model.h5', backbone_name="resnet50")
        if "new_license_plate.h5" in os.listdir(license_plate_model_dir):
            sp.call(f"mv {license_plate_model_dir}/new_license_plate_model.pb {license_plate_model_dir}/model.h5", shell=True)
            license_plate_model = models.load_model(license_plate_model_dir+'model.h5', backbone_name="resnet50")

        success, frame = vidcap.read()
        annotated_frame = detect_one_image(frame, face_model, license_plate_model)
        proc.stdin.write(annotated_frame.tostring())

if __name__ == '__main__':
    face_model = models.load_model(face_model_dir+'model.h5', backbone_name="resnet50")
    license_plate_model = models.load_model(license_plate_model_dir+'model.h5', backbone_name="resnet50")
    generate_frames(face_model, license_plate_model)


