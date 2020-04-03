import tensorflow as tf
import os
from PIL import Image
import xml.etree.ElementTree as ET
import random

extension = ".record"
xml_path = "../annotations/xmls"
image_path = "../images"


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


classtext_to_label = {
    "license_plate": 1,
    "face": 2
}


def create_tf_example(example):
    currentNameSplit = example.split('.')[0]
    currentImageName = currentNameSplit + '.png'

    with tf.io.gfile.GFile(os.path.join(image_path, '{}'.format(currentImageName)), 'rb') as fid:
        encoded_image_data = fid.read()

    filename = currentNameSplit.encode('utf8')
    image_format = b'png'

    tree = ET.parse(xml_path+"/"+example)
    root = tree.getroot()

    img = Image.open(image_path+"/"+currentImageName)
    width, height = img.size

    classes_text = [value.text.encode('utf8')
                    for value in root.findall("object/name")]
    classes = [classtext_to_label[value.text]
               for value in root.findall("object/name")]
    xmins = [int(value.text) for value in root.findall("object/bndbox/xmin")]
    ymins = [int(value.text) for value in root.findall("object/bndbox/ymin")]
    xmaxs = [int(value.text) for value in root.findall("object/bndbox/xmax")]
    ymaxs = [int(value.text) for value in root.findall("object/bndbox/ymax")]
    
    xmins = [value / width for value in xmins]
    ymins = [value / height for value in ymins]
    xmaxs = [value / width for value in xmaxs]
    ymaxs = [value / height for value in ymaxs]
   
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example


def make_tf_record(examples, name):
    writer = tf.io.TFRecordWriter(name+extension)
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
    writer.close()


def partition(examples, test_percent):
    testSet = []
    trainingSet = []
    howManyNumbers = int(round(test_percent*len(examples)))
    declineRandom = 0
    while True:
        if declineRandom == howManyNumbers:
            break
        randomIndex = random.randint(0, (len(examples)-1)-declineRandom)
        testSetTuple = examples[randomIndex]
        del examples[randomIndex]
        testSet.append(testSetTuple)

        declineRandom = declineRandom + 1
    trainingSet = examples[:]
    return (trainingSet), (testSet)


if __name__ == '__main__':
    examples = os.listdir(xml_path)
    train, val = partition(examples, test_percent=.2)
    make_tf_record(train, "train")
    make_tf_record(val, "val")
