<h1>How to train with Keras Retinanet</h1>


<h2>Recommended Folder Structure</h2>
The folder structure must be as shown. Thus, create any folders missing.

```
keras-retinanet
    |-- annotations (holds annotated xml files)
    |   |--face
    |   |--license_plate
    |-- csv (holds the csv files used for training)
    |   |--face
    |   |--license_plate
    |-- fine_tuned_model (holds the model we use for object detection)
    |   |--face
    |   |--license_plate
    |-- images (holds the images)
    |-- keras_retinanet (holds keras_retinanet stuff)
    |-- snapshots (holds snapshots created during training)
    ...
```

<h2>Preprocessing</h2>
First move all images into the images folder. Then move all xml files into the annotations folder. It is important that the xmls of annotated face data is moved to the annotations/face folder and the xmls of annotated license plate data is move to the annotations/license_plate folder.

```
$ mv path/to/images/* images/
$ mv path/to/face_xmls/*.xml annotations/face
$ mv path/to/license_plate_xmls/*.xml annotations/license_plate
```

**Create CSV files**

Before you can train your custom object detector, you must convert your data into csv files. Since we need to train as well as validate our model, the data set will be split into training (annotations.csv) and validation sets (val_annotations.csv). We’re going to use xml_to_csv.rb to convert our data set into csvs. Run the following two commands

```
# From keras-retinanet directory 
$ ruby xml_to_csv --annotation-path annotations/face --face
$ ruby xml_to_csv --annotation-path annotations/license_plate --license_plate
```

**Download COCO pretrained model**
```
# From keras-retinanet directory 
$ wget https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5
```


<h2>Training</h2>
Training must be done using tensorflow 1.x. Therefore, if you have tensorflow 2.x installed run the following

```
$ pip3 install tensorflow==1.15
```

<h3>Train using train.py</h3>
Training only requires one command, being the one below. However, it is IMPORTANT to change the arguments of steps and epochs to suit the needs. Steps must be set the the number of lines in the annotations.csv file.

**Train with license plates**


```
# From keras-retinanet directory
python3 keras_retinanet/bin/train.py --weights resnet50_coco_best_v2.1.0.h5 --steps 15 --epochs 1 csv csv/license_plate/annotations.csv csv/license_plate/classes.csv --val-annotations csv/license_plate/val_annotations.csv
```

**Train with faces**


```
# From keras-retinanet directory
python3 keras_retinanet/bin/train.py --weights resnet50_coco_best_v2.1.0.h5 --steps 15 --epochs 1 csv csv/face/annotations.csv csv/face/classes.csv --val-annotations csv/face/val_annotations.csv
```

**Enable tensorboard**
```
# From keras-retinanet
tensorboard --logdir=./
```

<h2>Model Export</h2>
Once you finish training your model, you can export your model to be used for inference. If you’ve been following the folder structure, use the following command:
Remember to change <highest_checkpoint_number> with the actual highest checkpoint number. The checkpoints are found in the train folder.

**Export license plate model**
```
# From keras-retinanet
$ python3 keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_<highest_checkpoint_number>.h5 fine_tuned_mode/license_plate/model.h5
```

**Export face model**
```
# From keras-retinanet
$ python3 keras_retinanet/bin/convert_model.py snapshots/resnet50_csv_<highest_checkpoint_number>.h5 fine_tuned_mode/face/model.h5
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