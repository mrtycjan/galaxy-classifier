import tensorflow as tf, sys
import os

imagePath = sys.argv[1]

imageRaw = tf.gfile.FastGFile(imagePath, 'rb').read()

labelLines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrainedLabels.txt")]

with tf.gfile.FastGFile("retrainedGraph.pb", 'rb') as f:
    graphDef = tf.compat.v1.GraphDef()
    graphDef.ParseFromString(f.read())
    _ = tf.import_graph_def(graphDef, name='')

with tf.compat.v1.Session() as session:
    softmaxTensor = session.graph.get_tensor_by_name('final_result:0')
    
    predictions = session.run(softmaxTensor, \
             {'DecodeJpeg/contents:0': imageRaw})
    
    top = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    os.system('clear')
    for nodeId in top:
        score = predictions[0][nodeId]
        print('%s (score = %.5f)' % (labelLines[nodeId], score))
