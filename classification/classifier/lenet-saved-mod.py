import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import io,transform

import cv2
from os import path, listdir, system
import numpy as np
import glob


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
print('tf版本：',tf.__version__)


def read_image(img_dir):
    label_dir = [path.join(img_dir, x) for x in listdir(img_dir) if path.isdir(path.join(img_dir, x))] 
    images = []
    labels = []
    for index,folder in enumerate(label_dir):
        for img in glob.glob(folder+'/*.bmp'):
            #print("reading the image:%s"%img)
            image = cv2.imread(img, 0)
            image = cv2.resize(image, (32, 32))
            # image = io.imread(img)
            # image = transform.resize(image,(28,28,1))
            images.append(image)
            labels.append(index)
    return np.asarray(images, dtype = np.float32), np.asarray(labels, dtype=np.int32) 

def load_data():
    img_dir = "/Users/macrosea/ws/work/watermark/classifier/tf4wm"
    train_data, train_label = read_image(path.join(img_dir, "train"))
    test_data, test_label = read_image(path.join(img_dir, "test"))
    return (train_data, train_label), (test_data, test_label)

def prepprocess(x,y):
    # 归一化
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, (32, 32, 1))
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=2)
    return x, y

#(train_image, train_label), (test_image, test_label) = keras.datasets.fashion_mnist.load_data()
(train_image, train_label), (test_image, test_label) = load_data()
train_image, train_label = shuffle(train_image, train_label)

print("数据维度(训练集)：", train_image.shape, train_label.shape)
print("数据维度(测试集)：", test_image.shape, test_label.shape)

train_data = tf.data.Dataset.from_tensor_slices((train_image, train_label))
train_data = train_data.map(prepprocess).batch(1024)
test_data = tf.data.Dataset.from_tensor_slices((test_image, test_label))
test_data = test_data.map(prepprocess).batch(1024)
#class_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
class_name = ['non', 'wm']

# plt.figure(figsize=(10, 10))
# for i in range(25):
# 	plt.subplot(5, 5, i + 1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_image[i], cmap=plt.cm.gray)
# 	plt.xlabel(class_name[train_label[i]])
# plt.show()


# #训练
# def train():
#     # 构建网络
network = keras.Sequential([
        keras.layers.Conv2D(6, (5,5), activation='relu', input_shape=(32,32, 1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(16, (5,5), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(84, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
])
network.build(input_shape=(None, 32, 32))
network.summary()

network.compile(optimizer=keras.optimizers.Adam(lr=0.005),
                loss = tf.losses.CategoricalCrossentropy(from_logits=True),
                # loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),      # 用这个不用tf.one_hot()
                metrics=['accuracy']
)
# 训练
history = network.fit(train_data, epochs=30, validation_data=test_data,validation_freq=1)
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.ylim([0.5,1])
plt.legend(loc='lower right')
plt.show()
tf.saved_model.save(network,'/tmp/model_save/wm/') 
print("保存模型成功")   


# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: network(x))
full_model = full_model.get_concrete_function(
tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
        logdir="/tmp/frozen_models",
        name="frozen_graph.pb",
        as_text=False)


