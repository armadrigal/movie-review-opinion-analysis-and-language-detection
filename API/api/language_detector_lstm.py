from fastapi import APIRouter
from typing import List
from DataModels import InputData, OutputDataLanguage
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

@router.post("/LanguageDetector/lstm")
async def language_detector_lstm(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_language_detector, main.stopwords_language_detector)
    y_pred = np.round(main.model_lstm(texts),2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataLanguage(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2], pt=y_pred[i,3], it=y_pred[i,4])
        responses.append(output_data)

    return responses