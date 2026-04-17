import pickle, os, random
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler
import xgboost
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename

# Suppress scikit-learn version warnings for old models
warnings.filterwarnings('ignore', category=UserWarning)

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'app_models')

def get_model(path):
    model = load_model(path, compile=False)
    return model

def pred(path):
    data = load_img(path, target_size=(224, 224, 3))
    data = np.asarray(data).reshape((-1, 224, 224, 3))
    data = data * 1.0 / 255
    model_path = os.path.join(MODELS_DIR, 'pneumonia_model.h5')
    predicted = np.round(get_model(model_path).predict(data)[0])[0]
    return predicted

def ValuePredictor(to_predict_list):
    if len(to_predict_list) == 15:
        page = 'kidney'
        model_path = os.path.join(MODELS_DIR, 'kidney_model.pkl')
        with open(model_path, 'rb') as f:
            kidney_model = pickle.load(f)
        pred = kidney_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 10:
        page = 'liver'
        model_path = os.path.join(MODELS_DIR, 'liver_model.pkl')
        with open(model_path, 'rb') as f:
            liver_model = pickle.load(f)
        pred = liver_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 11:
        page = 'heart'
        model_path = os.path.join(MODELS_DIR, 'heart_model.pkl')
        with open(model_path, 'rb') as f:
            heart_model = pickle.load(f)
        pred = heart_model.predict(np.array(to_predict_list).reshape(-1, len(to_predict_list)))
    elif len(to_predict_list) == 9:
        page = 'stroke'
        scaler_path = os.path.join(MODELS_DIR, 'avc_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            stroke_scaler = pickle.load(f)
        l1 = np.array(to_predict_list[2:]).reshape((-1, len(to_predict_list[2:]))).tolist()[0]
        l2 = stroke_scaler.transform(np.array(to_predict_list[0:2]).reshape((-1, 2))).tolist()[0]
        l = l2 + l1
        model_path = os.path.join(MODELS_DIR, 'avc_model.pkl')
        with open(model_path, 'rb') as f:
            stroke_model = pickle.load(f)
        pred = stroke_model.predict(np.array(l).reshape(-1, len(l)))
    elif len(to_predict_list) == 8:
        page = 'diabete'
        model_path = os.path.join(MODELS_DIR, 'diabete_model.pkl')
        with open(model_path, 'rb') as f:
            diabete_model = pickle.load(f)
        pred = diabete_model.predict(np.array(to_predict_list).reshape((-1, 8)))
        print(pred[0], page)
    return pred[0], page
