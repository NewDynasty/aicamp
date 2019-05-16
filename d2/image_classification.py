import os

import cv2
import matplotlib
import numpy as np
import tensorflow as tf
import time

from keras import Input, Model, optimizers
from keras.applications import InceptionV3, inception_v3
from keras.callbacks import EarlyStopping, TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense, regularizers, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.metrics import top_k_categorical_accuracy
from keras.utils.generic_utils import to_list
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


def print_log(s, br='\n', header='[info] ', with_time=False, decimal_format="0.3f",
              compare_last_log=False, compare_last_log_with_consuming=False):
    import time

    # 设置方法静态变量，记录初次调用打印方法时间戳
    if not hasattr(print_log, 'start'):
        print_log.start = time.time()

    if with_time:
        time_lapse = None
        # 与上一次打印日志耗时时间作比较
        if compare_last_log_with_consuming and hasattr(print_log, 'last_log_with_consuming_time'):
            time_lapse = time.time() - print_log.last_log_with_consuming_time
        elif compare_last_log and hasattr(print_log, 'last_log_time'):
            time_lapse = time.time() - print_log.last_log_time
        else:
            time_lapse = time.time() - print_log.start

        print_log.last_log_with_consuming_time = time.time()

        print(br + header + str(s) + ' [耗时：{}]'.format(format(time_lapse, decimal_format)))
    else:
        print(br + header + str(s))

    print_log.last_log_time = time.time()

class Chart(TensorBoard):
    def draw(self, outputs, num):
        logs = {}
        outputs = to_list(outputs)
        for l, o in zip(model.metrics_names, outputs):
            logs[l] = o

        for name, value in logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, num)
            self.writer.flush()

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

    print(model.summary())
    return model

def get_callbacks(model):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

    return [early_stopping]

def draw(H):
    # axis = np.arange(epochs) + 1
    # print(axis)
    plt.plot(H.history['loss'], label='train loss')
    plt.plot(H.history['val_loss'], label='valid loss')
    plt.legend()

    plt.figure()
    plt.plot(H.history['acc'], label='train acc')
    plt.plot(H.history['val_acc'], label='valid acc')
    plt.legend()

def id2label(valid, test_results):
    id2label = {val: key for key, val in valid.class_indices.items()}
    print('\n' + str(id2label))

    batch_y = None
    n = len(valid)
    for i in range(n):
        if batch_y is None:
            batch_y = valid[i][1]
        else:
            batch_y = np.vstack([batch_y, valid[i][1]])

    print_log(classification_report(
        batch_y.argmax(axis=1),
        test_results.argmax(axis=1),
        target_names=list(id2label.values())
    ))
    return id2label

def save_model(model, file_name=None, path=None):
    print_log('保存模型到本地...')

    if file_name is None:
        model_file_name = SETTINGS['model_file_name']
    else:
        model_file_name = file_name

    if path is not None and path != '':
        if not os.path.exists(path):
            os.makedirs(path)
        model_file_name = path + os.sep + model_file_name

    model.save(model_file_name)
    print_log('保存模型完成...文件名: {}'.format(model_file_name), with_time=True, compare_last_log=True)

def get_model():
    model_file_name = SETTINGS['model_file_name']
    if os.path.isfile(model_file_name):
        print_log('从文件中加载模型...')
        model = load_model(model_file_name)
        print_log('加载模型完成...', with_time=True, compare_last_log=True)
    else:
        print_log('创建末端全连通神经网络模型并加载优化算法...')
        model = create_new_nn(8, num_hidden_layers=1, keep_prob=.4, l2_lambda=1e-3)
        print_log('创建末端全连通神经网络模型并加载优化算法完成...', with_time=True, compare_last_log=True)

        # opt = optimizers.Adam(lr=.0001, beta_1=.95, beta_2=.999, epsilon=1e-8)
        opt = optimizers.SGD(lr=0.001, momentum=.9, nesterov=True)

        print_log('编译模型...')
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc', top_k_categorical_accuracy])
        print_log('编译模型完成...', with_time=True, compare_last_log=True)
    return model

def fit_on_batch(model, train_batches, valid_batches):
    print_log('训练开始...')

    epochs = SETTINGS['epoch']
    steps_per_epoch = len(train_batches) # 每epoch有多少batch
    global_steps = 0
    min_acc = 70
    eval_steps = 10 # 每eval_steps评估一次模型
    train_chart, valid_chart = get_charts(model)

    for epoch in range(epochs):
        print_log('Epoch {}/{}'.format(epoch + 1, epochs))

        for batch_num in range(steps_per_epoch):
            x_batch, y_batch = train_batches[batch_num]
            global_steps += 1

            outs = model.train_on_batch(x_batch, y_batch)

            if batch_num % 10 == 0:
                progress = batch_num / steps_per_epoch * 100
                print_log('steps: {}/{} | progress: {:.2f}%'.format(batch_num, steps_per_epoch, progress))

            if global_steps % eval_steps == 0:
                print_log('validating...')
                val_outs = model.evaluate_generator(
                    valid_batches,
                    len(valid_batches),
                    workers=4,
                    verbose=1
                )
                train_chart.draw(outs, global_steps // eval_steps)
                valid_chart.draw(outs, global_steps // eval_steps)

        val_acc = val_outs[2] * 100
        if val_acc > min_acc:
            print_log('saving intermediate model, rank-5 accuracy: {:.2f}%'.format(val_acc))
            save_model(model, 'tmp.h5', 'models')
            min_acc = val_acc

    save_model(model, 'final.h5', 'models')

    print_log('训练完成...', with_time=True, compare_last_log=True)
    # return H

def get_charts(model):
    train_chart = Chart(
        log_dir = os.path.join(SETTINGS['log_dir'], 'train'),
        histogram_freq = 0,
        write_graph = False,
        write_images = True
     )
    valid_chart = Chart(
        log_dir=os.path.join(SETTINGS['log_dir'], 'validation'),
        histogram_freq=0,
        write_graph=False,
        write_images=True
    )
    train_chart.set_model(model)
    valid_chart.set_model(model)
    return train_chart, valid_chart

def test_image(model, id2label):
    print_log('读取测试图片并开始预测...')
    test_image = cv2.imread('assets/test0.jpg')
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(test_image, (224, 224)) / 127.5 - 1

    pred = model.predict(np.array([input_image]))

    predicted_hero = id2label[pred.argmax()]
    print_log('模型认为这张图片为[{}].'.format(predicted_hero))
    display_image(test_image)

SETTINGS = {
    'model_file_name': 'hero_recognizer.h5',
    'log_dir': 'logs',
    'image_size': 224,
    'batch_size': 256,
    'shuffle': True,
    'epoch': 10
}

if __name__ == '__main__':
    print_log('创建图像读取生成器...')
    train, valid = get_image_loader(image_size=SETTINGS['image_size'], batch_size=SETTINGS['batch_size'], shuffle=SETTINGS['shuffle'])
    print_log('创建图像读取生成器完成...', with_time=True, compare_last_log=True)

    model = get_model()

    fit_on_batch(model, train, valid)

    draw(H)

    # test_results = model.predict(x_valid)

    eval_result = model.evaluate_generator(valid, steps=np.math.ceil(valid.samples/SETTINGS['batch_size']), verbose=1)
    print_log(str(eval_result))

    valid.reset()
    test_results = model.predict_generator(valid, steps=np.math.ceil(valid.samples/SETTINGS['batch_size']), verbose=1)

    id2label = id2label(valid, test_results)

    save_model(model)

    print_log('脚本执行完毕!', with_time=True)

    # test_image(model, id2label)
    input("\nPress <enter>")