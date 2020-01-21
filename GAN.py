import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# mnist data load
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# change image to float32 type
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

# 28x28 image -> 784 feature (flatten)
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

# normalinze
x_train, x_test = x_train / 255., x_test / 255.

# define function to show 8x8 grid mnist image
def plot(samples):
  fig = plt.figure(figsize=(8, 8))
  gs = gridspec.GridSpec(8, 8)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    plt.imshow(sample.reshape(28, 28))

  return fig

# set initial value
num_epoch = 100000
batch_size = 64
num_input = 28 * 28
num_latent_variable = 100
num_hidden = 128
learning_rate = 0.001

# shuffle data and get batch type
train_data = tf.data.Dataset.from_tensor_slices(x_train)
train_data = train_data.repeat().shuffle(60000).batch(batch_size)
train_data_iter = iter(train_data)

def random_normal_intializer(stddev):
  return tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=None)

# define generator
# latent variable -> generator image
class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()

    # 100 -> 128 -> 784
    self.hidden_layer_1 = tf.keras.layers.Dense(num_hidden,
                                                activation='relu',
                                                kernel_initializer=random_normal_intializer(5e-2),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.1))

    self.output_layer = tf.keras.layers.Dense(num_input,
                                              activation='sigmoid',
                                              kernel_initializer=random_normal_intializer(5e-2),
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))

  def call(self, x):
    hidden_layer = self.hidden_layer_1(x)
    generated_mnist_image = self.output_layer(hidden_layer)

    return generated_mnist_image

# define discriminator
# image -> predict value and logits
class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()

    # 784 -> 128 -> 1
    self.hidden_layer_1 = tf.keras.layers.Dense(num_hidden,
                                                activation='relu',
                                                kernel_initializer=random_normal_intializer(5e-2),
                                                bias_initializer=tf.keras.initializers.Constant(value=0.1))

    self.output_layer = tf.keras.layers.Dense(1,
                                              activation=None,
                                              kernel_initializer=random_normal_intializer(5e-2),
                                              bias_initializer=tf.keras.initializers.Constant(value=0.1))

  def call(self, x):
    hidden_layer = self.hidden_layer_1(x)
    logits = self.output_layer(hidden_layer)
    predicted_value = tf.nn.sigmoid(logits)

    return predicted_value, logits

# define loss function of generator
@tf.function
def generator_loss(D_fake_logits):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))         # log(D(G(z))

# define loss function of discriminator
@tf.function
def discriminator_loss(D_real_logits, D_fake_logits):
  d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))  # log(D(x))
  d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))  # log(1-D(G(z)))
  d_loss = d_loss_real + d_loss_fake  # log(D(x)) + log(1-D(G(z)))

  return d_loss

# run generator
Generator_model = Generator()

# run discriminator
Discriminator_model = Discriminator()

# define optimizer for generator and discriminator
discriminator_optimizer = tf.optimizers.Adam(learning_rate)
generator_optimizer = tf.optimizers.Adam(learning_rate)

# define function for discriminator optimizer
@tf.function
def d_train_step(discriminator_model, real_image, fake_image):
  with tf.GradientTape() as disc_tape:
    D_real, D_real_logits = discriminator_model(real_image)  # D(x)
    D_fake, D_fake_logits = discriminator_model(fake_image)  # D(G(z))
    loss = discriminator_loss(D_real_logits, D_fake_logits)
  gradients = disc_tape.gradient(loss, discriminator_model.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))

# define function for generator optimizer
@tf.function
def g_train_step(generator_model, discriminator_model, z):
  with tf.GradientTape() as gen_tape:
    G = generator_model(z)
    D_fake, D_fake_logits = discriminator_model(G)  # D(G(z))
    loss = generator_loss(D_fake_logits)
  gradients = gen_tape.gradient(loss, generator_model.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))

# create generated_output floder to save images
num_img = 0
if not os.path.exists('generated_output/'):
  os.makedirs('generated_output/')

# run num_epoch
for i in range(num_epoch):
  # get batch size mnist image
  batch_X = next(train_data_iter)

  # sampling batch size noise for input latent variable in uniform distribution
  batch_noise = np.random.uniform(-1., 1., [batch_size, 100]).astype('float32')

  # save image every 500 epoch
  if i % 500 == 0:
    samples = Generator_model(np.random.uniform(-1., 1., [64, 100]).astype('float32')).numpy()
    fig = plot(samples)
    plt.savefig('generated_output/%s.png' % str(num_img).zfill(3), bbox_inches='tight')
    num_img += 1
    plt.close(fig)

  # optimizer discriminatror and return loss function
  _, d_loss_print = d_train_step(Discriminator_model, batch_X, Generator_model(batch_noise)), discriminator_loss(Discriminator_model(batch_X)[1], Discriminator_model(Generator_model(batch_noise))[1])

  # optimizer generator and return loss funnction
  _, g_loss_print = g_train_step(Generator_model, Discriminator_model, batch_noise), generator_loss(Discriminator_model(Generator_model(batch_noise))[1])

  # print discriminator and generator loss function every 100 epoch
  if i % 100 == 0:
    print('epoch: %d, generator loss: %f, Discriminator d_loss: %f' % (i, g_loss_print, d_loss_print))