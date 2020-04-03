<h1>How to train with SSD MobileNet V2</h1>

**Setting up the environment**

Every time you start a new terminal window to work with the pre-trained models, it is important to compile Protobuf and change your PYTHONPATH. Run the following from your terminal:

1. ```cd tensorflow-ssd/research/```
2. ```protoc object_detection/protos/*.proto --python_out=.```
3. ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim```

**Preprocessing**
1. download ssd_mobilenet_v2_coco here and save its model checkpoint files (model.ckpt.meta, model.ckpt.index, model.ckpt.data-00000-of-00001) to our /checkpoints/ directory. http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

1. Move all images into the images folder
2. Move all xml files into the annotations/xmls folder
3. Update label_map.pbtxt
4. Copy all filenames (without extension) into trainval.txt
5. Create tf records by running the script in the tf_record folder

**Training**

6. Train using model_main.py