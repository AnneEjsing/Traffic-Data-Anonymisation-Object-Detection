import cv2
import numpy as np
import time
import subprocess as sp

model_path = 'model.h5'
model = models.load_model(model_path, backbone_name="resnet50") # https://github.com/fizyr/keras-retinanet/releases
#imgPath = "test.png"
video_path = "00001.mp4"
labels_to_names = {0: 'head', 1: 'license_plate'}

def detect_one_image(image):
  draw = image.copy()

  # preprocess image for network
  image = preprocess_image(image)
  image, scale = resize_image(image,540/5,960/5)

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

      color = label_color(label)
      b = box.astype(int)
      draw_box(draw, b, color=color)
      caption = "{} {:.3f}".format(labels_to_names[label], score)
      draw_caption(draw, b, caption)
  return draw

def generate_frames():
  # Load video file
  vidcap = cv2.VideoCapture(video_path)
  #pipeline_out = "appsrc ! videoconvert ! x264enc ! mpegtsmux ! queue ! hlssink target-duration=3 playlist-length=60"
  pipeline_out = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=2 byte-stream=true ! flvmux ! rtmpsink location='rtmp://0.0.0.0:1935/show/stream" # Create RTMP 
  fourcc = cv2.VideoWriter_fourcc(*'H264') # For HLS
  fps = vidcap.get(cv2.CAP_PROP_FPS)
  width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  writer = cv2.VideoWriter(pipeline_out, cv2.CAP_GSTREAMER, fourcc, fps, (width,height))
  success = True
#  command = ['ffmpeg',
#    '-y',
#    '-f', 'rawvideo',
#    '-vcodec','rawvideo',
#    '-pix_fmt', 'bgr24',
#    '-s', str(width)+'x'+str(height),
#    '-i', '-',
#    '-c:v', 'libx264',
#    '-pix_fmt', 'yuv420p',
#    '-preset', 'ultrafast',
#    '-f', 'flv',
#    'rtmp://localhost/show/stream']
#  proc = sp.Popen(command, stdin=sp.PIPE,shell=False)

  while success:
    success, frame = vidcap.read()
    annotated_frame = detect_one_image(frame)
    writer.write(frame)
    #proc.stdin.write(frame.tostring())

if __name__ == '__main__':
    generate_frames()


