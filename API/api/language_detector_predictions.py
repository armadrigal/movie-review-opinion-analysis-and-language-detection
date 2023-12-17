from process_data import ProcessData
import numpy as np
import main
from .data_models import OutputDataOpinion, OpinionPrediction, OutputDataOpinionAllModels

def predict(data, model):
    if model == "mlp":
        model_ = main.model_mlp
    elif model == "lstm":
        model_ = main.model_lstm
    elif model == "gru":
        model_ = main.model_gru
    texts = [item.text for item in data]
    print("se guargaron los textos en una lista")
    texts = ProcessData(texts, main.vocabulary_language_detector, main.stopwords_language_detector, language=language)
    print("se procesaron los textos")
    pred = model_.predict(texts).T[0]
    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataOpinion(id=item.id, opinion=pred[i])
        responses.append(output_data)
    print("se procede a enviar la respuesta")
    return responses