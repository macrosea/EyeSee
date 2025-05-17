import tensorflow as tf
import glob
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.python.ops import summary_ops_v2
import cv2
from os import path, listdir, system
import numpy as np

class LeNet(tf.keras.Model):
    def __init__(self, classes_num):
        super(LeNet, self).__init__()

        self.conv1 = Conv2D(32, 5, activation=tf.nn.relu)
        self.pool1 = MaxPool2D(2, strides=2)
        self.conv2 = Conv2D(64, 5, activation=tf.nn.relu)
        self.pool2 = MaxPool2D(2, strides=2)

        self.flatten = Flatten()
        self.fc1 = Dense(1024, activation=tf.nn.relu)
        self.dropout = Dropout(0.5)
        self.out = Dense(classes_num)

    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)

        # if not training:
        #     x = tf.nn.softmax(x)

        return x

def cross_entropy_loss(y_pred, y_true):
    if y_true.ndim == 1:
        y_true = tf.cast(y_true, tf.int64)
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    else:
        cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(cost)


def accuracy(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.int64) if y_pred.ndim == 1 else tf.argmax(y_pred, 1)
    y_true = tf.cast(y_true, tf.int64) if y_true.ndim == 1 else tf.argmax(y_true, 1)

    correct_prediction = tf.equal(y_pred, y_true)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

def read_image(img_dir):
    label_dir = [path.join(img_dir, x) for x in listdir(img_dir) if path.isdir(path.join(img_dir, x))] 
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.bmp'):
            #print("reading the image:%s"%img)
            image = cv2.imread(img, 0)
            image = cv2.resize(image, (28, 28))
            # image = io.imread(img)
            # image = transform.resize(image,(w,h,c))
            images.append(image)
            labels.append(index)
    return np.asarray(images, dtype = np.float32), np.asarray(labels, dtype=np.int32) 

def load_data():
    img_dir = "/Users/macrosea/ws/work/watermark/classifier/tf4wm"
    train_data, train_label = read_image(path.join(img_dir, "train"))
    test_data, test_label = read_image(path.join(img_dir, "test"))
    return (train_data, train_label), (test_data, test_label)

def train():
    system("rm -rf /tmp/watermark/saved ; rm -rf /tmp/watermark/ckpoint")
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)) / 255.
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255.

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat(3).shuffle(5000).batch(32).prefetch(1)
    
    net = LeNet(2)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=net)  

    optimizer = tf.optimizers.Adam(0.001)

    log_dir = "/tmp/tensorboard"
    summary_writer = tf.summary.create_file_writer(log_dir)  
    tf.summary.trace_on(profiler=True) 

    for step, (x, y_true) in enumerate(train_data, 1):
        with tf.GradientTape() as tape:
            y_pred = net(x, training=True)
            loss = cross_entropy_loss(y_pred, y_true)

        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        if step % 10 == 0:
            path = checkpoint.save('/tmp/watermark/ckpoint/model.ckpt') 
            print("model saved to %s" % path)
            acc = accuracy(y_pred, y_true)
            print(f"step:{step},loss:{loss:05f},acc:{acc:05f}")

        with summary_writer.as_default():                         
            tf.summary.scalar("loss", loss, step=step)  
            
 

    with summary_writer.as_default():
        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)  

    tf.saved_model.save(net, "/tmp/watermark/saved/1", signatures={"call": net.call})



def test():
    net = LeNet(2)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=net)      
    checkpoint.restore(tf.train.latest_checkpoint('/tmp/watermark/ckpoint'))    
    _, (x_test, y_test) = load_data()
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255.
    y_pred = np.argmax(net.predict(x_test), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == y_test) / len(y_test)))

def test_serv():
    batch_size = 1
    model = tf.saved_model.load("/tmp/watermark/saved/1")
    res = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #data_loader = MNISTLoader()
    _, (x_test, y_test) = load_data()
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255.
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(len(x_test) // batch_size)
    for batch_index in range(num_batches - 2):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.call(x_test[start_index: end_index])
        sparse_categorical_accuracy.update_state(y_true=y_test[start_index: end_index], y_pred=y_pred)
    print("test accuracy: %f" % sparse_categorical_accuracy.result())

if __name__ == '__main__':
    # for dev in tf.config.experimental.list_physical_devices('GPU'):
    #     tf.config.experimental.set_memory_growth(dev, True)
    # if args.mode == 'train':
    #     train()
    # if args.mode == 'test':
    #      test()
    train()
    test_serv()