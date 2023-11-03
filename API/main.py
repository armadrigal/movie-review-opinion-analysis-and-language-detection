from fastapi import FastAPI
import tensorflow as tf
import pickle
from api.language_detector_mlp import router as language_detector_mlp
from api.language_detector_lstm import router as language_detector_lstm
app = FastAPI()

#Language Detector Models
model_mlp = tf.keras.models.load_model("./Models/model_mlp.h5")
model_lstm = tf.keras.models.load_model("./Models/model_lstm.h5")

#Vocabularies 
with open("./vocabulary/vocabulary_language_detector", "rb") as file:
    vocabulary_language_detector = pickle.load(file)

#Stopwords
with open("./vocabulary/stopwords_language_detector", "rb") as file:
    stopwords_language_detector = pickle.load(file)

app.include_router(language_detector_mlp)
app.include_router(language_detector_lstm)


