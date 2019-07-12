## Dependecies
from __future__ import division, print_function
import os
import sys
import argparse
import numpy as np

#Image 
import cv2
from sklearn.preprocessing import normalize

# Keras
import keras
from keras.layers import *
from keras.optimizers import SGD
from keras.models import load_model, Model
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from keras import backend as K

# Flask utils
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

# Config 

parser = argparse.ArgumentParser()
parser.add_argument("-w1", "--width", help="Target Image Width", type=int, default=256)
parser.add_argument("-h1", "--height", help="Target Image Height", type=int, default=256)
parser.add_argument("-c1", "--channel", help="Target Image Channel", type=int, default=1)
parser.add_argument("-p", "--path", help="Best Model Location Path", type=str, default="models")
parser.add_argument("-s", "--save", help="Save Uploaded Image", type=bool, default=False)
parser.add_argument("--port", help="WSGIServer Port ID", type=int, default=5000)
args = parser.parse_args()

SHAPE              = (args.width, args.height, args.channel)
MODEL_SAVE_PATH    = args.path
SAVE_LOADED_IMAGES = args.save

# Metrics
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+K.epsilon()))

# SE block
def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def create_model():
    
    dropRate = 0.3
    
    init = Input(SHAPE)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(init) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same')(x) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x1 = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (3, 3), activation=None, padding='same')(x1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, (5, 5), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x2 = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3, 3), activation=None, padding='same')(x2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(128, (2, 2), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = squeeze_excite_block(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x3 = MaxPooling2D((2,2))(x)
    
    ginp1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x1)
    ginp2 = UpSampling2D(size=(4, 4), interpolation='bilinear')(x2)
    ginp3 = UpSampling2D(size=(8, 8), interpolation='bilinear')(x3)
    
    concat = Concatenate()([ginp1, ginp2, ginp3]) 
    gap = GlobalAveragePooling2D()(concat)
    
    x = Dense(256, activation=None)(gap)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropRate)(x)
    
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dense(4, activation='softmax')(x)
    
    model = Model(init, x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-3, momentum=0.9), metrics=['acc', precision, recall, f1])
    return model

model = create_model()
print(model.summary())
model.load_weights(os.path.join(MODEL_SAVE_PATH,"heysaw_fold_"+str(1)+".h5")) # Loading best model.
print('Model loaded. Check http://localhost:{}/'.format(args.port))


def model_predict(img_path, model):
    img = np.array(cv2.imread(img_path))
    img = cv2.resize(img, SHAPE[:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = normalize(img)
    img = np.expand_dims(img, axis=2)
    
    prediction = model.predict(np.expand_dims(img, axis=0), batch_size=1)
    return prediction

# Threshold predictions
def threshold_arr(array):
    new_arr = []
    for ix, val in enumerate(array):
        loc = np.array(val).argmax(axis=0)
        k = list(np.zeros((len(val)), dtype=np.float16))
        k[loc]=1
        new_arr.append(k)
        
    return np.array(new_arr, dtype=np.float16)


os.chdir("deploy/")
# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        pred_class = threshold_arr(preds)[0]
        if pred_class[0] == 1:
            result = "Choroidal Neovascularization (" + str(preds[0][0]) + ")"
        elif pred_class[1] == 1:
           result = "Diabetic Macular Edema (" + str(preds[0][1]) + ")"
        elif pred_class[2] == 1:
           result = "DRUSEN (" + str(preds[0][2]) + ")"
        elif pred_class[3] == 1:
           result =  "NORMAL (" + str(preds[0][3]) + ")"

        if not SAVE_LOADED_IMAGES:
        	os.remove(file_path)

        return result
    return None

	
if __name__ == '__main__':
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
    sys.exit()