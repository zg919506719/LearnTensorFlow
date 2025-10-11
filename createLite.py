# TensorFlow and tf.keras
import os

import tensorflow as tf

import keras
# from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# A helper function that returns 'red'/'black' depending on if its two input
# parameter matches or not.
def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_tflite_model(tflite_model):
  # Initialize TFLite interpreter using the model.
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_tensor_index = interpreter.get_input_details()[0]["index"]
  output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_tensor_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy


if __name__ == '__main__':
    print(tf.__version__)
    # Keras provides a handy API to download the MNIST dataset, and split them into
    # "train" dataset and "test" dataset.
    mnist = keras.datasets.mnist
    # 训练集 测试集
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    # Show the first 25 images in the training dataset.
    # 创建10×10英寸的画布


    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.gray)
    #     plt.xlabel(train_labels[i])
    # plt.show()


    # 训练 TensorFlow 模型来对数字图像进行分类
    # 接下来，我们使用 Keras API 构建一个 TensorFlow 模型，并在 MNIST 数据集上进行训练。
    # 训练完成后，我们的模型将能够对数字图像进行分类。
    # 我们的模型以28px x 28px 的灰度图像作为输入，并输出长度为 10 的浮点数组，
    # 表示图像为 0 到 9 的数字的概率。
    # 这里我们使用一个简单的卷积神经网络，这是计算机视觉领域的一种常见技术
    # Define the model architecture
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(10)
    ])
    # Define how to train the model
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the digit classification model
    model.fit(train_images, train_labels, epochs=5)
    # model.summary()

    # Evaluate the model using all images in the test dataset.
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    # Predict the labels of digit images in our test dataset.
    predictions = model.predict(test_images)

    # As the model output 10 float representing the probability of the input image
    # being a digit from 0 to 9, we need to find the largest probability value
    # to find out which digit the model predicts to be most likely in the image.
    prediction_digits = np.argmax(predictions, axis=1)

    # Then plot 100 random test images and their predicted labels.
    # If a prediction result is different from the label provided label in "test"
    # dataset, we will highlight it in red color.
    plt.figure(figsize=(18, 18))
    for i in range(100):
        ax = plt.subplot(10, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_index = random.randint(0, len(prediction_digits))
        plt.imshow(test_images[image_index], cmap=plt.cm.gray)
        ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index], \
                                                 test_labels[image_index]))
        plt.xlabel('Predicted: %d' % prediction_digits[image_index])
    plt.show()

    # Convert Keras model to TF Lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()
    # Show model size in KBs.
    float_model_size = len(tflite_float_model) / 1024
    print('Float model size = %dKBs.' % float_model_size)

    # Re-convert the model to TF Lite using quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 关键设置：指定操作符版本兼容性
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # 使用TFLite内置操作符
        tf.lite.OpsSet.SELECT_TF_OPS  # 对于不支持的操作符使用TensorFlow操作符
    ]

    # 设置目标兼容版本
    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_lower_tensor_list_ops = False

    tflite_quantized_model = converter.convert()

    # Show model size in KBs.
    quantized_model_size = len(tflite_quantized_model) / 1024
    print('Quantized model size = %dKBs,' % quantized_model_size)
    print('which is about %d%% of the float model size.' \
          % (quantized_model_size * 100 / float_model_size))

    # 相对于当前工作目录
    save_dir = "./saved_tflite_models"  # 当前目录下的文件夹
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "mnist.tflite")
    with open(save_path, 'wb') as f:
        f.write(tflite_quantized_model)

    print(f"模型已保存到: {save_path}")

    # Evaluate the TF Lite float model. You'll find that its accurary is identical
    # to the original TF (Keras) model because they are essentially the same model
    # stored in different format.
    float_accuracy = evaluate_tflite_model(tflite_float_model)
    print('Float model accuracy = %.4f' % float_accuracy)

    # Evalualte the TF Lite quantized model.
    # Don't be surprised if you see quantized model accuracy is higher than
    # the original float model. It happens sometimes :)
    quantized_accuracy = evaluate_tflite_model(tflite_quantized_model)
    print('Quantized model accuracy = %.4f' % quantized_accuracy)
    print('Accuracy drop = %.4f' % (float_accuracy - quantized_accuracy))