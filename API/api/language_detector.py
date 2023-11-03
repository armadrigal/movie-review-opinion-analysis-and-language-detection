from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

class InputTexts(BaseModel):
    id: int
    text: str

class InputData(BaseModel):
    model: str
    texts: List[InputTexts]

class OutputData(BaseModel):
    id: int
    es: float
    en: float
    fr: float

class LanguagePrediction(BaseModel):
    es: float
    en: float
    fr: float

class OutputDataAllModels(BaseModel):
    id: int
    mlp: LanguagePrediction
    lstm: LanguagePrediction
    gru: LanguagePrediction

@router.post("/LanguageDetector/")
async def language_detector(data: InputData):

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
            output_data = OutputData(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2])
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
                output_data = OutputData(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2])
                responses.append(output_data)

            return responses

        elif data.model == 'all':

            responses = []
            
            for i, item in enumerate(data.texts):
            
                y_pred_mlp = np.round(y_pred_mlp, 2)
                mlp_pred = LanguagePrediction(es=y_pred_mlp[i,0], en=y_pred_mlp[i,1], fr=y_pred_mlp[i,2])
                y_pred_lstm = np.round(y_pred_lstm, 2)
                lstm_pred = LanguagePrediction(es=y_pred_lstm[i,0], en=y_pred_lstm[i,1], fr=y_pred_lstm[i,2])
                y_pred_gru = np.round(y_pred_gru, 2)
                gru_pred = LanguagePrediction(es=y_pred_gru[i,0], en=y_pred_gru[i,1], fr=y_pred_gru[i,2])

                output_data = OutputDataAllModels(id=item.id, mlp=mlp_pred, lstm=lstm_pred, gru=gru_pred)

                responses.append(output_data)

            return responses

            





