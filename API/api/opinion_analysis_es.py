from process_data import ProcessData
import numpy as np
import main
from .data_models import OutputDataOpinion, OpinionPrediction, OutputDataOpinionAllModels

def predict(data, model):
    if model == "lstm":
        model_ = main.model_lstm_es
    elif model == "gru":
        model_ = main.model_gru_es
    elif model == "cnn":
        model_ = main.model_cnn_es
    elif model == "attention":
        model_ = main.model_attention_es
    elif model == "lstm+attention":
        model_ = main.model_lstm_attention_es
    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_es, main.stopwords_es, language='es')
    pred = model_.predict(texts).T[0]
    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataOpinion(id=item.id, opinion=pred[i])
        responses.append(output_data)
    return responses

def predict_all_models(data, model):
    models = []
    models.append(main.model_lstm_es)
    models.append(main.model_gru_es)
    models.append(main.model_cnn_es)
    models.append(main.model_attention_es)
    models.append(main.model_lstm_attention_es)
    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_es, main.stopwords_es, language='es')
    predictions = []
    for model_ in models:
        predictions.append(model_.predict(texts).T[0]) 
    responses = []
    if model == "all":
        for i, item in enumerate(data):
            pred = OpinionPrediction(
                lstm=predictions[0][i],
                gru=predictions[1][i],
                cnn=predictions[2][i],
                attention=predictions[3][i],
                lstm_attention=predictions[4][i],
            )
            output_data = OutputDataOpinionAllModels(id=item.id, opinion=pred)
            responses.append(output_data)
    elif model == "avg":
        pred = (predictions[0] + predictions[1] + predictions[2] + predictions[3] + predictions[4])/5
        for i, item in enumerate(data):
            output_data = OutputDataOpinion(id=item.id, opinion=pred[i])
            responses.append(output_data)
    elif model == "logit":
        predictions_ = []
        for pred in predictions:
            predictions_.append(np.expand_dims(pred, axis=1))
        model_ = main.model_logit_es
        pred = model_.predict(np.concatenate(predictions_, axis=1)).T[0]
        for i, item in enumerate(data):
            output_data = OutputDataOpinion(id=item.id, opinion=pred[i])
            responses.append(output_data)
    return responses
