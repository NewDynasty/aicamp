from datetime import datetime

from keras.applications import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import cv2
import os


class NeuralStyle:
    def __init__(self, settings):
        self.settings = settings
        w, h = load_img(self.settings['input_path']).size
        self.dims = h, w

        self.content = self.preprocess(settings['input_path'])
        self.style = self.preprocess(settings['style_path'])
        self.content = K.variable(self.content)
        self.style = K.variable(self.style)

        self.output = K.placeholder((1, *self.dims, 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        print("[info] loading network...")
        self.model = self.settings['net'](include_top=False, input_tensor=self.input)

        layer_map = {layer.name: layer.output for layer in self.model.layers}

        content_features = layer_map[self.settings['content_layer']]
        style_features = content_features[0, :, :, :]
        output_features = content_features[2, :, :, :]

        content_loss = self.feature_recon_loss(style_features, output_features)
        content_loss *= self.settings['content_weight']

        style_loss = K.variable(0.)
        weight = 1.0 / len(self.settings['style_layers'])

        for layer in self.settings['style_layers']:
            style_output = layer_map[layer]
            style_features = style_output[1, :, :, :]
            output_features = style_output[2, :, :, :]

            tmp = self.style_recon_loss(style_features, output_features)
            style_loss += weight * tmp

        style_loss *= self.settings['style_weight']
        tv_loss = self.settings['tv_weight'] * self.tv_loss(self.output)

        total_loss = content_loss + style_loss + tv_loss

        grads = K.gradients(total_loss, self.output)
        outputs = [total_loss]
        outputs += grads

        self.loss_and_grads = K.function([self.output], outputs)

    def preprocess(self, path):
        image = load_img(path, target_size=self.dims)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)

    def deprocess(self, image):
        image = image.reshape([*self.dims, 3])
        image[:, :, 0] += 103.939
        image[:, :, 1] += 116.779
        image[:, :, 2] += 123.680

        image = np.clip(image, 0, 255).astype('uint8')
        return image

    def gram_mat(self, x):
        features = K.permute_dimensions(x, (2, 0, 1))
        features = K.batch_flatten(features)
        return K.dot(features, K.transpose(features))

    def feature_recon_loss(self, style_features, output_features):
        return K.sum(K.square(output_features - style_features))

    def style_recon_loss(self, style_features, output_features):
        a = self.gram_mat(style_features)
        g = self.gram_mat(output_features)

        scale = 1.0 / (2 * 3 * np.prod(self.dims)) ** 2
        loss = scale * K.sum(K.square(g - a))
        return loss

    def tv_loss(self, x):
        h, w = self.dims
        a = K.square(x[:, :h - 1, :w - 1, :] - x[:, 1:, :w - 1, :])
        b = K.square(x[:, :h - 1, :w - 1, :] - x[:, :h - 1, 1:, :])
        loss = K.sum(K.pow(a + b, 1.25))
        return loss

    def transfer(self, max_evals=20):
        x = np.random.uniform(0, 255, (1, *self.dims, 3)) - 128

        for i in range(0, self.settings['iterations']):
            print("[info] starting iteration #{}/{}...".format(i + 1, self.settings['iterations']))
            x, loss, _ = fmin_l_bfgs_b(
                self.loss,
                x.flatten(),
                fprime=self.grads,
                maxfun=max_evals,
            )
            print('[info] end of iteration {}, loss: {:.2e}'.format(i + 1, loss))
            image = self.deprocess(x.copy())
            p = os.path.sep.join([
                self.settings['output_path'], "iter_{}.jpg".format(i)
            ])
            cv2.imwrite(p, image)

    def loss(self, x):
        x = x.reshape([1, *self.dims, 3])
        tick = datetime.now()
        loss_value = self.loss_and_grads([x])[0]
        tock = datetime.now()
        print("compute loss takes: {}".format(tock - tick))
        return loss_value

    def grads(self, x):
        x = x.reshape([1, *self.dims, 3])
        tick = datetime.now()
        output = self.loss_and_grads([x])
        tock = datetime.now()
        print("compute gradients takes: {}".format(tock - tick))
        return output[1].flatten().astype('float64')


if __name__ == '__main__':
    SETTINGS = {
        'input_path': "/Users/ethan/Pictures/证件照/profile_sm.jpg",
        "style_path": '/Users/ethan/Pictures/starry_night.jpg',
        'output_path': "/Users/ethan/Pictures/trans",

        'net': VGG19,
        'content_layer': 'block4_conv2',
        'style_layers': [
            'block1_conv1', 'block2_conv1', 'block3_conv1',
            'block4_conv1', 'block5_conv1'
        ],
        'content_weight': 0.02,
        'style_weight': 1.0,
        'tv_weight': 0.5,  # total-variation loss weight

        'iterations': 20,
    }
    ns = NeuralStyle(SETTINGS)
    ns.transfer()