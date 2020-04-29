<h1>How to train with SSD MobileNet V2</h1>

<h2>Setting up the environment</h2>

***VERY IMPORTANT***

Every time you start a new terminal window to work with the pre-trained models, it is important to compile Protobuf and change your PYTHONPATH. Run the following from your terminal:

```
$ cd tensorflow-ssd/research/
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

<h2>Recommended Folder Structure</h2>
The folder structure must be as shown. Thus, create any folders missing.

```
tensorflow-ssd
    |-- annotations
    |   |--xmls
    |-- images
    |-- checkpoints
    |-- tf_record
    |-- research
    ...
```

<h2>Preprocessing</h2>
First move all images into the images folder. Then move all xml files into the annotations/xmls folder. Easily done with the following
```
$ mv path/to/images/* images/
$ mv path/to/xmls/*.xml annotations/xmls
```

**Download model from model zoo**

Download ssd_mobilenet_v2_coco here and save its model checkpoint files (model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001) to our /checkpoints/ directory. http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

**Create Label Map (.pbtxt)**

Classes need to be listed in the label map. The structure of the label map is like the following. Note, that the ids are 1-indexed. If this file is not present in the annotations folder, then create it.
```
item {
    id: 1
    name: 'license_plate'
}
item {
    id: 2
    name: 'face'
}
```

**Create TFRecord (.record)**

Before you can train your custom object detector, you must convert your data into the TFRecord format. Since we need to train as well as validate our model, the data set will be split into training (train.record) and validation sets (val.record). The purpose of training set is straight forward - it is the set of examples the model learns from. The validation set is a set of examples used DURING TRAINING to iteratively assess model accuracy. We’re going to use create_tf_record.py to convert our data set into train.record and val.record.

This script supports sharding. If the number of images exceeds it is beneficial to split the tfrecords into more. The number of shards is given as an input parameter. This script is preconfigured to do 80–20 train-val split. Execute it by running:

```
# From tensorflow-ssd directory 
$ python3 tf_record/create_tf_record.py
```

It has three flags being -f, -l and -s. 
-f creates records for face data (default: false)
-l creates records for license plate data (default: false)
-fl / -lf creates records for both.
-s indicates the number of shards to create (defualt: 1)

<h2>Training</h2>
Training must be done using tensorflow 1.x. Therefore, if you have tensorflow 2.x installed run the following

```
$ pip3 install tensorflow==1.15
```

<h3>Train using model_main.py (from research folder)</h3>

TRAIN WITH LICENSE PLATES

```
python3 object_detection/model_main.py --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v2_coco.config --model_dir=../train/license --logtostderr
```

TRAIN WITH FACES

```
python3 object_detection/model_main.py --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_coco.config --model_dir=../train/face --logtostderr
```


Enable tensorboard (from either train/face or train/license folder)
```
tensorboard --logdir=./
```

<h2>Model Export</h2>
Once you finish training your model, you can export your model to be used for inference. If you’ve been following the folder structure, use the following command:

```
$ mkdir fine_tuned_model

$ python3 research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix  train/model.ckpt-<the_highest_checkpoint_number>  --output_directory fine_tuned_model
```

WRITE DOCUMENTATION ON FACE EXPORT

python3 research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix  face_model/model.ckpt --output_directory fine_tuned_model_face

  # Add global step to the graph.
  slim.get_or_create_global_step()


<h2>Run Detection</h2>
Must be run with tensorflow 2. As training must be done with tensorflow 1.x upgrade tensorflow

```
$ pip3 install tensorflow==2.1
```


```
# From tensorflow-ssd directory

$ python3 research/object_detection/detection.py
```