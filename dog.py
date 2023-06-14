from flask import Flask ,render_template , request
import pandas as pd
import numpy as np
import pickle
import json
from flask_cors import CORS
import time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse
pd.set_option('display.max_columns', None)

#Importing required libraries
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

import tensorflow as tf
# from keras import Sequential
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam, SGD
# from keras.callbacks import ReduceLROnPlateau
# from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
# from keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img


app = Flask(__name__)
CORS(app)



@app.route("/")
def home():
     return "<h1 style='color:blue'>Hello pycharm!</h1>"


@app.route('/test',methods = ['GET','POST'])
def test1():
      
      
      labels = pd.read_csv('C:/Users/rumma/OneDrive/Desktop/dog_predict/labels.csv')\
      #Create list of alphabetically sorted labels.
      classes = sorted(list(set(labels['breed'])))
      n_classes = len(classes)
      # print('Total unique breed {}'.format(n_classes))

      #Map each label string to an integer label.
      class_to_num = dict(zip(classes, range(n_classes)))

      labels = labels.head(1)

      image_labels = labels[:]['breed']

      input_shape = (331,331,3)
      target_size = input_shape
      images = np.zeros([len(labels[:]), target_size[0], target_size[1], target_size[2]],dtype=np.uint8) 

      y = np.zeros([len(labels[:]),1],dtype = np.uint8)
    
      for ix, image_name in enumerate(tqdm(labels[:]['id'].values)):
              img_dir = os.path.join('C:/Users/rumma/OneDrive/Desktop/dog_predict/dog-test-2', image_name + '.jpg')
              print(img_dir)
              img = load_img(img_dir, target_size = target_size)
              images[ix] = img
              del img

      def get_features(model_name, model_preprocessor, input_size, data):
              input_layer = Input(input_size)
              preprocessor = Lambda(model_preprocessor)(input_layer)
              base_model = model_name(weights='imagenet', include_top=False,
                                      input_shape=input_size)(preprocessor)
              avg = GlobalAveragePooling2D()(base_model)
              feature_extractor = Model(inputs = input_layer, outputs = avg)
              
              #Extract feature.
              feature_maps = feature_extractor.predict(data, verbose=1)
              print('Feature maps shape: ', feature_maps.shape)
              return feature_maps
      
      img_size = (331,331,3)
      # Extract features using InceptionV3 
      from keras.applications.inception_v3 import InceptionV3, preprocess_input
      inception_preprocessor = preprocess_input
      inception_features = get_features(InceptionV3,
                                        inception_preprocessor,
                                        img_size, images)

      # print('Inception feature maps shape', inception_features.shape)
      
      model = tf.keras.models.load_model('C:/Users/rumma/OneDrive/Desktop/dog_predict/model.h5')

      dog_index = np.argmax(model.predict(inception_features),axis=1)
        
      for i in class_to_num:
          if class_to_num[i]==dog_index[0]:
            return i
           

      # return str(np.argmax(model.predict(inception_features),axis=1))






if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
