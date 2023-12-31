from fastapi import FastAPI
import tensorflow as tf
import pickle
from api.language_detector import router as language_detector
from api.language_detector_mlp import router as language_detector_mlp
from api.language_detector_lstm import router as language_detector_lstm
from api.language_detector_gru import router as language_detector_gru
from api.opinion_es_lstm import router as opinion_es_lstm
from api.opinion_es_gru import router as opinion_es_gru
from api.opinion_es_cnn import router as opinion_es_cnn
from api.opinion_es_attention import router as opinion_es_attention
from api.opinion_es_lstm_attention import router as opinion_es_lstm_attention
from api.opinion_en_lstm import router as opinion_en_lstm
from api.opinion_en_gru import router as opinion_en_gru
from api.opinion_en_cnn import router as opinion_en_cnn
from api.opinion_en_attention import router as opinion_en_attention
from api.opinion_en_lstm_attention import router as opinion_en_lstm_attention
from api.opinion_fr_lstm import router as opinion_fr_lstm
from api.opinion_fr_gru import router as opinion_fr_gru
from api.opinion_fr_cnn import router as opinion_fr_cnn

app = FastAPI()

#Language Detector Models
model_mlp = tf.keras.models.load_model("./Models/model_mlp.h5")
model_lstm = tf.keras.models.load_model("./Models/model_lstm.h5")
model_gru = tf.keras.models.load_model("./Models/model_gru.h5")

#Opinion Analysis Models
model_lstm_es = tf.keras.models.load_model("./Models/model_lstm_es.h5")
model_gru_es = tf.keras.models.load_model("./Models/model_gru_es.h5")
model_cnn_es = tf.keras.models.load_model("./Models/model_cnn_es.h5")
model_attention_es = tf.keras.models.load_model("./Models/model_attention_es.h5")
model_lstm_attention_es = tf.keras.models.load_model("./Models/model_lstm_attention_es.h5")
model_lstm_en = tf.keras.models.load_model("./Models/model_lstm_en.h5")
model_gru_en = tf.keras.models.load_model("./Models/model_gru_en.h5")
model_cnn_en = tf.keras.models.load_model("./Models/model_cnn_en.h5")
model_attention_en = tf.keras.models.load_model("./Models/model_attention_en.h5")
model_lstm_attention_en = tf.keras.models.load_model("./Models/model_lstm_attention_en.h5")
model_lstm_fr = tf.keras.models.load_model("./Models/model_lstm_fr.h5")
model_gru_fr = tf.keras.models.load_model("./Models/model_gru_fr.h5")
model_cnn_fr = tf.keras.models.load_model("./Models/model_cnn_fr.h5")

#Vocabularies 
with open("./vocabulary/vocabulary_language_detector", "rb") as file:
    vocabulary_language_detector = pickle.load(file)
with open("./vocabulary/vocabulary_es", "rb") as file:
    vocabulary_es = pickle.load(file)
with open("./vocabulary/vocabulary_en", "rb") as file:
    vocabulary_en = pickle.load(file)
with open("./vocabulary/vocabulary_fr", "rb") as file:
    vocabulary_fr = pickle.load(file)

#Stopwords
with open("./vocabulary/stopwords_language_detector", "rb") as file:
    stopwords_language_detector = pickle.load(file)
with open("./vocabulary/stopwords_es", "rb") as file:
    stopwords_es = pickle.load(file)
with open("./vocabulary/stopwords_en", "rb") as file:
    stopwords_en = pickle.load(file)
with open("./vocabulary/stopwords_fr", "rb") as file:
    stopwords_fr = pickle.load(file)

app.include_router(language_detector)
app.include_router(language_detector_mlp)
app.include_router(language_detector_lstm)
app.include_router(language_detector_gru)
app.include_router(opinion_es_lstm)
app.include_router(opinion_es_gru)
app.include_router(opinion_es_cnn)
app.include_router(opinion_es_attention)
app.include_router(opinion_es_lstm_attention)
app.include_router(opinion_en_lstm)
app.include_router(opinion_en_gru)
app.include_router(opinion_en_cnn)
app.include_router(opinion_en_attention)
app.include_router(opinion_en_lstm_attention)
app.include_router(opinion_fr_lstm)
app.include_router(opinion_fr_gru)
app.include_router(opinion_fr_cnn)


