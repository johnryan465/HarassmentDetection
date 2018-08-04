import tensorflow as tf

saver = tf.train.import_meta_graph("/tmp/keras_model.ckpt.meta")
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "/tmp/keras_model.ckpt")

for op in graph.get_operations():
    print(op.name)

output_node_names = []
output_node_names.append("Sigmoid")  # Specify the real node name
output_graph_def = tf.graph_util.convert_variables_to_constants(
    sess,
    input_graph_def,
    output_node_names
)


output_file = "model_file.pb"
with tf.gfile.GFile(output_file, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
