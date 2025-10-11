import keras
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import shutil

# from tensorflow import keras
import matplotlib.pyplot as plt


def create_model():
  model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model


# 从上面的 25 张图片中我们可以看到，数字的大小大致相同，并且位于图像的中心。让我们验证一下这个假设在 MNIST 数据集上是否成立
def digit_area(mnist_image):
    # Remove the color axes
    mnist_image = np.squeeze(mnist_image, axis=2)

    # Extract the list of columns that contain at least 1 pixel from the digit
    x_nonzero = np.nonzero(np.amax(mnist_image, 0))
    x_min = np.min(x_nonzero)
    x_max = np.max(x_nonzero)

    # Extract the list of rows that contain at least 1 pixel from the digit
    y_nonzero = np.nonzero(np.amax(mnist_image, 1))
    y_min = np.min(y_nonzero)
    y_max = np.max(y_nonzero)

    return [x_min, x_max, y_min, y_max]

if __name__ == '__main__':
    print(tf.__version__)
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Add a color dimension to the images in "train" and "validate" dataset to
    # leverage Keras's data augmentation utilities later.
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # base_model = create_model()
    # base_model.fit(
    #     train_images,
    #     train_labels,
    #     epochs=5,
    #     validation_data=(test_images, test_labels)
    # )

    # Show the first 25 images in the training dataset.
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(np.squeeze(train_images[i], axis=2), cmap=plt.cm.gray)
    #     plt.xlabel(train_labels[i])
    # plt.show()

    # Calculate the area containing the digit across MNIST dataset
    # digit_area_rows = []
    # for image in train_images:
    #     digit_area_row = digit_area(image)
    #     digit_area_rows.append(digit_area_row)
    # digit_area_df = pd.DataFrame(
    #     digit_area_rows,
    #     columns=['x_min', 'x_max', 'y_min', 'y_max']
    # )
    # digit_area_df.hist()
    # # 显示直方图
    # plt.show()

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.25,
        zoom_range=0.2
    )

    # Generate augmented data from MNIST dataset
    train_generator = datagen.flow(train_images, train_labels)
    test_generator = datagen.flow(test_images, test_labels)
    augmented_images, augmented_labels = next(train_generator)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.squeeze(augmented_images[i], axis=2), cmap=plt.cm.gray)
        plt.xlabel('Label: %d' % augmented_labels[i])
    # 显示直方图
    plt.show()

    improved_model = create_model()
    improved_model.fit(train_generator, epochs=5, validation_data=test_generator)
    test_result=improved_model.evaluate(test_generator)
    print(test_result)

    # Convert Keras model to TF Lite format and quantize.
    converter = tf.lite.TFLiteConverter.from_keras_model(improved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    # Save the quantized model to file to the Downloads directory
    f = open('mnist.tflite', "wb")
    f.write(tflite_quantized_model)
    f.close()

    # Download the digit classification model

    # 复制文件到当前目录
    shutil.copy('mnist.tflite', './update')
    print("文件已保存到当前目录")