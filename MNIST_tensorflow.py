import os

import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = '1'
tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#Build the model
class CustomizedCNN(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool_1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc_1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc_2 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(tf.expand_dims(inputs, axis=-1), tf.float32)
        conv_1 = self.conv_1(inputs)
        pool_1 = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool_1)
        flatten = self.flatten(conv_2)
        if training:
            flatten = tf.nn.dropout(flatten, 0.25)
        fc_1 = self.fc_1(flatten)
        if training:
            fc_1 = tf.nn.dropout(fc_1, 0.25)
        logits = self.fc_2(fc_1)
        return logits

model = CustomizedCNN()
model.build(input_shape=(None, 28, 28))
model.summary()


model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

#train model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), batch_size=128)


#plot accuracy 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1])
plt.legend(loc='lower right')

#get loss
test_loss, test_acc = model.evaluate(test_images,  test_labels)
print(test_acc)