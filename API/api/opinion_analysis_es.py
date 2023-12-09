from process_data import ProcessData
import main
from .data_models import OutputDataOpinion

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

def predict_all_models(texts, model):
    models = []
    models.append(main.model_lstm_es)
    models.append(main.model_gru_es)
    models.append(main.model_cnn_es)
    models.append(main.model_attention_es)
    models.append(main.model_lstm_attention_es)