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

@router.post("/OpinionAnalysis/es/attention", summary="Opinion Analysis, Language: Spanish, Model: Attention")
async def opinion_es_attention(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_es, main.stopwords_es, language='es')
    y_pred = np.round(main.model_attention_es(texts), 2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputData(id=item.id, opinion=y_pred[i,0])
        responses.append(output_data)

    return responses