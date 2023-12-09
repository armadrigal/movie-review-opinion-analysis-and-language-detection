from process_data import ProcessData
import main
from .data_models import OutputDataOpinion

def predict(data, model):
    if model == "lstm":
        model_ = main.model_lstm_en
    elif model == "gru":
        model_ = main.model_gru_en
    elif model == "cnn":
        model_ = main.model_cnn_en
    elif model == "attention":
        model_ = main.model_attention_en
    elif model == "lstm+attention":
        model_ = main.model_lstm_attention_en
    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_en, main.stopwords_en, language='en')
    pred = model_.predict(texts).T[0]
    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataOpinion(id=item.id, opinion=pred[i])
        responses.append(output_data)
    return responses
