import os
import sys
import wget

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
K.clear_session()

from minio import Minio
from minio.error import ResponseError


class Classifier():

    def __init__(self):
        minioClient = Minio('140.114.79.72:9000',
                  access_key='nmsl',
                  secret_key='nmslinnthu',
                  secure=False)
        try:
            data = minioClient.get_object('umc', 'model.h5')
            with open('model.h5', 'wb') as file_data:
                for d in data.stream(32*1024):
                    file_data.write(d)
        except ResponseError as err:
            print(err)
        # K.clear_session()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        set_session(self.sess)
        # load the trained model
        self.net = load_model('./model.h5')
        self.graph = tf.get_default_graph()
        print("model loaded")
        pass

    def evaluate(self, file):
        cls_list = ['cats', 'dogs']
        img = image.load_img(file, target_size=(299,299))
        if img is None:
            return 'unknown'
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        pred = self.net.predict(x)[0]
        top_inds = pred.argsort()[::-1][:5]
        return cls_list[0]

    def predict(self, X, **kwargs):
        
#        Return a prediction.
#        Parameters
#        ----------
#        X : array-like
#        feature_names : array of feature names (optional)
        result = []
        print(X)
        for x in X:
            print(x)
            filename = wget.download(x)
            print(filename)
            cls_list = ['cats', 'dogs']
            img = image.load_img(filename, target_size=(299,299))
            if img is None:
                return 'unknown'
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            with self.graph.as_default():
                set_session(self.sess)
                pred = self.net.predict(x)[0]
            top_inds = pred.argsort()[::-1][:5]
            result.append(cls_list[0])
            # result.append(self.evaluate(filename))
        print("Predict called - will run identity function")
        return result

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--url", "-url", help="image file url")
    args = parser.parse_args()
    print(args)
#     files = get_files(args.path)
    cls_list = ['cats', 'dogs']
    
    classifier = Classifier()
    print(classifier.predict([args.url])[0])
    
    # loop through all files and make predictions
#     for f in files:
#     img = image.load_img(f, target_size=(299,299))
#     if img is None:
#         continue
#     x = image.img_to_array(img)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#     pred = net.predict(x)[0]
#     top_inds = pred.argsort()[::-1][:5]
#     print(f)
#     for i in top_inds:
#         print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
