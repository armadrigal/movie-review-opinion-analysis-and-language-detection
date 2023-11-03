from fastapi import FastAPI
import tensorflow as tf
import pickle
from api.language_detector_mlp import router as language_detector_mlp
from api.language_detector_lstm import router as language_detector_lstm
from api.language_detector_gru import router as language_detector_gru
from api.language_detector import router as language_detector
from api.opinion_en_lstm import router as opinion_en_lstm

app = FastAPI()

#Language Detector Models
model_mlp = tf.keras.models.load_model("./Models/model_mlp.h5")
model_lstm = tf.keras.models.load_model("./Models/model_lstm.h5")
model_gru = tf.keras.models.load_model("./Models/model_gru.h5")

#Opinion Analysis Models
model_lstm_en = tf.keras.models.load_model("./Models/model_lstm_en.h5")

#Vocabularies 
with open("./vocabulary/vocabulary_language_detector", "rb") as file:
    vocabulary_language_detector = pickle.load(file)
with open("./vocabulary/vocabulary_en", "rb") as file:
    vocabulary_en = pickle.load(file)

#Stopwords
with open("./vocabulary/stopwords_language_detector", "rb") as file:
    stopwords_language_detector = pickle.load(file)
with open("./vocabulary/stopwords_en", "rb") as file:
    stopwords_en = pickle.load(file)

app.include_router(language_detector_mlp)
app.include_router(language_detector_lstm)
app.include_router(language_detector_gru)
app.include_router(language_detector)
app.include_router(opinion_en_lstm)


