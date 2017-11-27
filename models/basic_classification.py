import tensorflow as tf

# Network Parameters
n_hidden_1 = 10        # 1st layer number of features
n_hidden_2 = 5         # 2nd layer number of features
n_input = total_words  # Words in vocab
n_classes = 3          # Categories: graphics, space and baseball

def multilayer_perceptron(input_tensor, weights, biases):

    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1_activation = tf.nn.relu(layer_1_addition)

	# Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2_activation = tf.nn.relu(layer_2_addition)

	# Output layer with linear activation
    out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']

	return out_layer_addition

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

learning_rate = 0.001

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss
entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor)
loss = tf.reduce_mean(entropy_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)