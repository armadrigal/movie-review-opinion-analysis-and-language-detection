from fastapi import APIRouter
from DataModels import InputDataCustomLanguage, OutputDataLanguage, LanguagePrediction, OutputDataAllModelsLanguage
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

@router.post("/LanguageDetector/")
async def language_detector(data: InputDataCustomLanguage):

    texts = [item.text for item in data.texts]
    texts = ProcessData(texts, main.vocabulary_language_detector, main.stopwords_language_detector)
    
    if data.model == 'mlp' or data.model == 'lstm' or data.model == 'gru':
        if data.model == 'mlp':
            model = main.model_mlp
        elif data.model == 'lstm':
            model = main.model_lstm
        elif data.model == 'gru':
            model = main.model_gru
        y_pred = np.round(model.predict(texts), 2)

        responses = []
        for i, item in enumerate(data.texts):
            output_data = OutputDataLanguage(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2], pt=y_pred[i,3], it=y_pred[i,4])
            responses.append(output_data)

        return responses

    elif data.model == 'all' or data.model == 'average':
        y_pred_mlp = main.model_mlp.predict(texts)
        y_pred_lstm = main.model_lstm.predict(texts)
        y_pred_gru = main.model_gru.predict(texts)

        if data.model == 'average':
            y_pred = np.round((1/3)*(y_pred_mlp + y_pred_lstm + y_pred_gru), 2)

            responses = []
            for i, item in enumerate(data.texts):
                output_data = OutputDataLanguage(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2], pt=y_pred[i,3], it=y_pred[i,4])
                responses.append(output_data)

            return responses

        elif data.model == 'all':

            responses = []
            
            for i, item in enumerate(data.texts):
            
                y_pred_mlp = np.round(y_pred_mlp, 2)
                mlp_pred = LanguagePrediction(es=y_pred_mlp[i,0], en=y_pred_mlp[i,1], fr=y_pred_mlp[i,2], pt=y_pred_mlp[i,3], it=y_pred_mlp[i,4])
                y_pred_lstm = np.round(y_pred_lstm, 2)
                lstm_pred = LanguagePrediction(es=y_pred_lstm[i,0], en=y_pred_lstm[i,1], fr=y_pred_lstm[i,2], pt=y_pred_lstm[i,3], it=y_pred_lstm[i,4])
                y_pred_gru = np.round(y_pred_gru, 2)
                gru_pred = LanguagePrediction(es=y_pred_gru[i,0], en=y_pred_gru[i,1], fr=y_pred_gru[i,2], pt=y_pred_gru[i,3], it=y_pred_gru[i,4])

                output_data = OutputDataAllModelsLanguage(id=item.id, mlp=mlp_pred, lstm=lstm_pred, gru=gru_pred)

                responses.append(output_data)

            return responses

            





