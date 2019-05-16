import os

import cv2
import matplotlib
import numpy as np
import time
from keras import Input, Model, optimizers
from keras.applications import InceptionV3, inception_v3
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

def normalize_image(images):
    return images / 127.5 - 1

def restore_image(images):
    return (images + 1) / 2

def display_image(*images, col=None):
    if col is None:
        col = len(images)
    row = np.math.ceil(len(images) / col)
    plt.figure(figsize=(20, int(20 * col / row)))
    for i, image in enumerate(images):
        image = image.squeeze()
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.show()

def get_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    # image_root = "/Users/ethan/datasets/marvel"
    # image_root = '/Users/dynasty/Documents/workspace/code_python/aicamp/d2/marvel'
    # image_root = '/home/dynasty/code/aicamp/d2/marvel'
    image_root = 'K:\\Work\\code\\real_code\\code_python\\aicamp\\d2\\marvel'

    train_data_gen = ImageDataGenerator(
        # rescale=1 / 255.0,
        preprocessing_function=inception_v3.preprocess_input,
        horizontal_flip=True,
        rotation_range=10
    )
    valid_data_gen = ImageDataGenerator(
        # rescale=1 / 255.0,
        preprocessing_function=inception_v3.preprocess_input
    )
    train_loader = train_data_gen.flow_from_directory(
        os.path.join(image_root, 'train'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    valid_loader = valid_data_gen.flow_from_directory(
        os.path.join(image_root, 'valid'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader, valid_loader

def get_feature_extractor():
    base = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    btnk = base.get_layer('mixed8').output
    features = GlobalAveragePooling2D()(btnk)
    return Model(base.input, features)

def create_nn(input_size, num_classes):
    inputs = Input(shape=(input_size,))
    x = Dropout(.6)(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.6)(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(.01))(x)
    return Model(inputs, out)

def create_new_nn(num_classes, num_hidden_layers=0, keep_prob=.5, l2_lambda=1e-3):
    base = InceptionV3(include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # btnk = base.get_layer('mixed3').output
    # features = GlobalAveragePooling2D()(btnk)
    x = Dropout(1 - keep_prob)(base.output)

    for _ in range(num_hidden_layers):
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(1 - keep_prob)(x)

    # out = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(.1))(x)
    out = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_lambda))(x)
    model = Model(base.input, out)
    
    for layer in model.layers:
        layer.trainable = False
        if layer.name == 'mixed3':
            break

    # print(model.summary())
    return model

if __name__ == '__main__':
    start_time = time.time()
    batch_size = 256
    train, valid = get_image_loader(image_size=224, batch_size=batch_size, shuffle=True)

    print("\n[info] 创建末端全连通神经网络模型并加载优化算法...")
    # model = create_nn(x_train.shape[1], y_train.shape[1])
    model = create_new_nn(8, num_hidden_layers=1, keep_prob=.4, l2_lambda=1e-3)

    print('\n[info] 创建模型完成， 加载优化算法。。。')
    # opt = optimizers.Adam(lr=.0001, beta_1=.95, beta_2=.999, epsilon=1e-8)
    opt = optimizers.SGD(lr=0.001, momentum=.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

    train_start_time = time.time()
    print("\n[info] 开始训练...")
    epochs = 25
    # H = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_valid, y_valid), verbose=1)
    H = model.fit_generator(train, epochs=epochs, verbose=1, callbacks=[early_stopping],
                            validation_data=valid, validation_steps=np.math.ceil(valid.samples/batch_size),
                            steps_per_epoch=np.math.ceil(train.samples/batch_size))

    print("\n[info] 训练完成! 训练耗时：" + str(time.time() - train_start_time) + " 开始检测模型性能...")

    # axis = np.arange(epochs) + 1
    # print(axis)
    plt.plot(H.history['loss'], label='train loss')
    plt.plot(H.history['val_loss'], label='valid loss')
    plt.legend()

    plt.figure()
    plt.plot(H.history['acc'], label='train acc')
    plt.plot(H.history['val_acc'], label='valid acc')
    plt.legend()

    # test_results = model.predict(x_valid)

    eval_result = model.evaluate_generator(valid, steps=np.math.ceil(valid.samples/batch_size), verbose=1)
    print('\n' + str(eval_result))

    valid.reset()
    test_results = model.predict_generator(valid, steps=np.math.ceil(valid.samples/batch_size), verbose=1)


    id2label = {val: key for key, val in valid.class_indices.items()}
    print('\n' + str(id2label))

    batch_y = None
    n = len(valid)
    for i in range(n):
        if batch_y is None:
            batch_y = valid[i][1]
        else:
            batch_y = np.vstack([batch_y, valid[i][1]])

    print(classification_report(
        batch_y.argmax(axis=1),
        test_results.argmax(axis=1),
        target_names=list(id2label.values())
    ))
    #
    # print("\n[info] 创建完整模型...")
    # bottleneck = feature_extractor.output
    # out = model(bottleneck)
    # hero_recognizer = Model(feature_extractor.input, out)
    # print("\n[info] 创建成功! 保存模型到本地...")

    # model_file_name = "hero_recognizer.h5"
    # model.save(model_file_name)
    # print("\n[info] 保存完成, 文件名: {}".format(model_file_name))
    #
    # print('\n[info] 从文件中加载模型...')
    # new_model = load_model('hero_recognizer.h5')
    print("\n[info] 加载完成, 读取测试图片并开始预测...")
    test_image = cv2.imread('assets/test0.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(test_image, (224, 224)) / 127.5 - 1

    pred = model.predict(np.array([input_image]))

    predicted_hero = id2label[pred.argmax()]
    print("\n[info] 模型认为这张图片为[{}].".format(predicted_hero))
    display_image(test_image)

    print('\n[info] 脚本执行完毕! 总耗时:' + str(time.time() - start_time))
