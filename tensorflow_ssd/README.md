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
    |-- annotations (holds annotated xml files)
    |   |--xmls
    |   |   |--face
    |   |   |--license_plate
    |-- checkpoints (holds the checkpoints from which we start training)
    |   |--face
    |   |--license_plate
    |-- fine_tuned_model (holds the model we use for object detection)
    |   |--face
    |   |--license_plate
    |-- images (holds the images)
    |-- research (holds tensorflow stuff)
    |-- tf_record (holds the tfrecords describing the training data)
    |   |--face
    |   |--license_plate
    |-- train (holds checkpoints created during training)
    |   |--face
    |   |--license_plate
    ...
```

<h2>Preprocessing</h2>
First move all images into the images folder. Then move all xml files into the annotations/xmls folder. It is important that the xmls of annotated face data is moved to the annotations/xmls/face folder and the xmls of annotated license plate data is move to the annotations/xmls/license_plate folder.

```
$ mv path/to/images/* images/
$ mv path/to/face_xmls/*.xml annotations/xmls/face
$ mv path/to/license_plate_xmls/*.xml annotations/xmls/license_plate
```

**Download model from model zoo**

* Download ssd_mobilenet_v2_coco: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
* Save the checkpoint files (model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001) to the /checkpoints/license_plate directory. 

**Download face detection model**

* Download the frozen_inference_graph.pb from: https://github.com/yeephycho/tensorflow-face-detection/tree/master/model.
* Place it in the /checkpoitns/face directory
* Convert the frozen inference graph to an ssd model using
```
# From tensorflow-ssd directory 
$ python3 frozen_to_model
```
* Make sure that the /checkpoitns/face directory now contains the same files as the /checkpoints/license_plate directory. 


**Create TFRecord (.record)**

Before you can train your custom object detector, you must convert your data into the TFRecord format. Since we need to train as well as validate our model, the data set will be split into training (train.record) and validation sets (val.record). We’re going to use create_tf_record.py to convert our data set into train.record and val.record.

This script supports sharding. If the number of images exceeds it is beneficial to split the tfrecords into more. The number of shards is given as an input parameter. This script is preconfigured to do 80–20 train-val split. Execute it by running:

```
# From tensorflow-ssd directory 
$ python3 tf_record/create_tf_record.py
```

It has three flags being -f, -l and -s. 

* -f creates records for face data (default: false)
* -l creates records for license plate data (default: false)
* -fl / -lf creates records for both.
* -s indicates the number of shards to create (defualt: 1)

<h2>Training</h2>
Training must be done using tensorflow 1.x. Therefore, if you have tensorflow 2.x installed run the following

```
$ pip3 install tensorflow==1.15
```

<h3>Train using model_main.py (from research folder)</h3>
Training only requires one command, being the follwing:

**Train with license plates**


```
# From tensorflow-ssd/research directory
python3 object_detection/model_main.py --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v2_coco.config --model_dir=../train/license_plate --logtostderr
```

**Train with faces**


```
# From tensorflow-ssd/research directory
python3 object_detection/model_main.py --pipeline_config_path=object_detection/samples/configs/ssd_mobilenet_v1_coco.config --model_dir=../train/face --logtostderr
```

**Enable tensorboard**
```
# From tensorflow-ssd
tensorboard --logdir=./
```

<h2>Model Export</h2>
Once you finish training your model, you can export your model to be used for inference. If you’ve been following the folder structure, use the following command:
Remember to change <highest_checkpoint_number> with the actual highest checkpoint number. The checkpoints are found in the train folder.

**Export license plate model**
```
# From tensorflow-ssd
$ python3 research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix  train/license_plate/model.ckpt-<highest_checkpoint_number>  --output_directory fine_tuned_model/license_plate
```

**Export face model**
```
# From tensorflow-ssd
$ python3 research/object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix  train/face/model.ckpt-<highest_checkpoint_number>  --output_directory fine_tuned_model/face
```

<h2>Run Detection</h2>
Must be run with tensorflow 2. As training must be done with tensorflow 1.x upgrade tensorflow

```
$ pip3 install tensorflow==2.1
```


```
# From tensorflow-ssd directory
$ python3 detection.py
```