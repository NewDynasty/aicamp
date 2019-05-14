from keras.engine.saving import load_model
import cv2
import os
from keras_applications import inception_v3
from keras_preprocessing.image import ImageDataGenerator
import numpy as np


def get_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    # image_root = "/Users/ethan/datasets/marvel"
    image_root = '/Users/dynasty/Documents/workspace/code_python/aicamp/d2/marvel'
    # image_root = '/home/dynasty/code/aicamp/d2/marvel'

    data_gen = ImageDataGenerator(
        # rescale=1 / 255.0,
        preprocessing_function=inception_v3.preprocess_input
    )
    train_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'train'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    valid_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'valid'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader, valid_loader

def get_test_image_loader(image_size, batch_size, color_mode='rgb', shuffle=True):
    image_root = '/Users/dynasty/Documents/workspace/code_python/aicamp/d2/marvel'
    # image_root = '/home/dynasty/code/aicamp/d2/marvel'

    data_gen = ImageDataGenerator(
        # rescale=1 / 255.0,
        preprocessing_function=inception_v3.preprocess_input
    )
    test_loader = data_gen.flow_from_directory(
        os.path.join(image_root, 'train'),
        color_mode=color_mode,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return test_loader

if __name__ == '__main__':
    batch_size = 32
    train, valid = get_image_loader(image_size=224, batch_size=batch_size, shuffle=True)
    test = get_test_image_loader(image_size=224, batch_size=batch_size, shuffle=True)

    id2label = {val: key for key, val in valid.class_indices.items()}
    print(id2label)

    print('\n[info] 从文件中加载模型...')
    new_model = load_model('hero_recognizer.h5')
    print("\n[info] 加载完成, 读取测试图片并开始预测...")

    for img in test:
        print(img)
        test_image = cv2.imread('assets/test0.jpg')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(test_image, (224, 224)) / 127.5 - 1

        pred = new_model.predict(np.array([input_image]))

        predicted_hero = id2label[pred.argmax()]
        print("\n[info] 模型认为这张图片为[{}].".format(predicted_hero))

    # display_image(test_image)
    print('\n[info] 脚本执行完毕!')