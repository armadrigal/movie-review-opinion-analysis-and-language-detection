from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

class InputData(BaseModel):
    id: int
    text: str

class OutputData(BaseModel):
    id: int
    es: float
    en: float
    fr: float

@router.post("/LanguageDetector/gru")
async def language_detector_gru(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_language_detector, main.stopwords_language_detector)
    y_pred = np.round(main.model_gru(texts),2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputData(id=item.id, es=y_pred[i,0], en=y_pred[i,1], fr=y_pred[i,2])
        responses.append(output_data)

    return responses