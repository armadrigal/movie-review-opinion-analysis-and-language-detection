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
    opinion: float

@router.post("/OpinionAnalysis/fr/lstm")
async def opinion_fr_lstm(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_fr, main.stopwords_fr, language='fr')
    y_pred = np.round(main.model_lstm_fr(texts), 2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputData(id=item.id, opinion=y_pred[i,0])
        responses.append(output_data)

    return responses