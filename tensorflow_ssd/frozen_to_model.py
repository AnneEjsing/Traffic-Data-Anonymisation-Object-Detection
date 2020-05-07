import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def protobuf_to_checkpoint_conversion(pb_model, ckpt_dir):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def,name='')

    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    dummy = np.random.random((1, 512, 512, 3))
    
    with graph.as_default():
        config = tf.ConfigProto()
        with tf.Session(graph=graph, config=config) as sess:
            constant_ops = [op for op in graph.get_operations() if op.type == "Const"]
            vars_dict = {}
            ass = []
            for constant_op in constant_ops:
                name = constant_op.name
                const = constant_op.outputs[0]
                shape = const.shape
                var = tf.get_variable(name, shape, dtype=const.dtype, initializer=tf.zeros_initializer())
                vars_dict[name] = var

            print('INFO:Initializing variables')
            init = tf.global_variables_initializer()
            sess.run(init)

            print('INFO: Loading vars')
            for constant_op in tqdm(constant_ops):
                name = constant_op.name
                if 'FeatureExtractor' in name or 'BoxPredictor' in name:
                    const = constant_op.outputs[0]
                    shape = const.shape
                    var = vars_dict[name]
                    var.load(sess.run(const, feed_dict={image_tensor:dummy}), sess)
        
            saver = tf.train.Saver(var_list=vars_dict)
            ckpt_path = os.path.join(ckpt_dir, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=0)
    return graph, vars_dict

if __name__ == "__main__":
    protobuf_to_checkpoint_conversion("checkpoints/face/frozen_inference_graph_face.pb", "checkpoints/face")
    