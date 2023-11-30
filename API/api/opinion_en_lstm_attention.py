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

@router.post("/OpinionAnalysis/en/lstmattention")
async def opinion_en_lstm_attention(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_en, main.stopwords_en, language='en')
    y_pred = np.round(main.model_lstm_attention_en(texts), 2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputData(id=item.id, opinion=round(y_pred[i,0],2))
        responses.append(output_data)

    return responses