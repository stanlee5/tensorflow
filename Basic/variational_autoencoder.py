from datetime import datetime
import time as T
import tensorflow as tf
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt

# raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
from numpy.distutils.misc_util import yellow_text

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

startTime = T.time()
def strTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

np.random.seed(0)
tf.set_random_seed(0)

def xavier_init(fan_in, fan_out, constant=1):
    # stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    # Xavier initialization: one of init method
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=np.float32)

class VariationalAutoencoder(object):
    # VAE with sklearn-like interface using Tensorflow

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create Autoencoder network
        self._create_network()

        # Define Loss function
        self._create_loss_optimizer()

        # Initializing tf variables
        init = tf.global_variables_initializer()   # for>=0.12
        #init = tf.initialize_all_variables()      # for old

        # Launch the session
        config = tf.ConfigProto()
        #config.log_device_placement=True
        #config.gpu_options.allocator_type = "BFC"
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencoder's w, b
        network_weights = self._initialize_weights(**self.network_architecture)

        # Recognition network to determine mean, (log) var of Gaussian of latent code
        self.z_mean, self.z_log_sigma_sq = self._recognition_network(network_weights["weights_recog"],
                                                                     network_weights["biases_recog"])
        # Draw on sample z from gaussian
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)

        # z = mu + sigma*epsilon (Reparametrization trick)
        self.z = tf.add(self.z_mean,
                        tf.mul(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoiulli of reconstructed input
        self.x_reconstr_mean = self._generator_network(network_weights["weights_gener"],
                                                       network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog']={
            'h1' : tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2' : tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean' : tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma' : tf.Variable(xavier_init(n_hidden_recog_2, n_z))
        }
        all_weights['biases_recog']={
            'b1' : tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2' : tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean' : tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma' : tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }

        all_weights['weights_gener']={
            'h1' : tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2' : tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean' : tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma' : tf.Variable(xavier_init(n_hidden_gener_2, n_input))
        }
        all_weights['biases_gener'] = {
            'b1' : tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2' : tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean' : tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma' : tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }
        return all_weights

    # Generate probabilistic encoder
    # Maps input -> normal distribution in latent space
    def _recognition_network(self, weights, biases):
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        return (z_mean, z_log_sigma)

    # Generate probabilistic decoder
    # Maps points in latent space onto a bernoulli in data space
    def _generator_network(self, weights, biases):
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # 1) reconstruction loss (negative log prob)
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                                       + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)

        # 2) latent loss (KL divergence)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        # Average over batch
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    # Train model on mini-batch, return cost
    def partial_fit(self, x):
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict = {self.x: x})
        return cost

    # Transform data into latent space
    def transform(self, x):
        # Sample from gaussian
        return self.sess.run(self.z_mean, feed_dict={self.x: x})

    # Generate data by sampling from latent space
    def generate(self, z_mu=None):
        # If z_mu isnt None, data from this point generated
        # Otherwise, from prior in latent space
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    # Use VAE to reconstruct given data
    def reconstruct(self, x):
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: x})


def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)

        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "Cost=", "{:.9f}".format(avg_cost), strTime())
    return vae



def show_testImage(vae):
    x_sample = mnist.test.next_batch(100)[0]
    x_reconstruct = vae.reconstruct(x_sample)
    plt.figure(figsize=(8,12))
    for i in range(5):
        plt.subplot(5, 2, 2*i+1)
        plt.imshow(x_sample[i].reshape(28,28), vmin=0, vmax=1, cmap="gray")
        plt.title("VAE input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i+2)
        plt.imshow(x_reconstruct[i].reshape(28,28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()

def show_latent(vae):
    x_sample, y_sample = mnist.test.next_batch(5000)
    z_mu = vae.transform(x_sample)
    plt.figure(figsize=(8,6))
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
    plt.colorbar()
    plt.grid()
    plt.show()


def show_continuous(vae):
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = np.array([[xi, yi]]*vae.batch_size)
            x_mean = vae.generate(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28,28)
    plt.figure(figsize=(8,10))
    xi, yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, orgin="upper", cmap="gray")
    plt.tight_layout()
    plt.show()

network_architecture = dict(n_hidden_recog_1=500,
                            n_hidden_recog_2=500,
                            n_hidden_gener_1=500,
                            n_hidden_gener_2=500,
                            n_input=784,    # 28*28
                            n_z=64)

vae = train(network_architecture, training_epochs=20)

network_architecture_2d = dict(n_hidden_recog_1=500,
                            n_hidden_recog_2=500,
                            n_hidden_gener_1=500,
                            n_hidden_gener_2=500,
                            n_input=784,    # 28*28
                            n_z=2)
vae2d = train(network_architecture_2d, training_epochs=1)



#show_testImage(vae=vae)
#show_latent(vae=vae2d)
show_continuous(vae=vae)

