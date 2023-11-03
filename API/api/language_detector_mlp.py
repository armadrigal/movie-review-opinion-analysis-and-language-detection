from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from process_data import ProcessData
import main

router = APIRouter()

class InputData(BaseModel):
    id: int
    text: str

class OutputData(BaseModel):
    id: int
    es: float
    en: float

@router.post("/language_detector_mlp")
async def language_detector_mlp(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_language_detector, main.stopwords_language_detector)
    y_pred = main.model_mlp(texts)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputData(id=item.id, es=y_pred[i,0], en=1-y_pred[i,0])
        responses.append(output_data)

    return responses