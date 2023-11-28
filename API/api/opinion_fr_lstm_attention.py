from fastapi import APIRouter
from typing import List
from DataModels import InputData, OutputDataOpinion
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

@router.post("/OpinionAnalysis/fr/lstmattention")
async def opinion_it_lstm_attention(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_fr, main.stopwords_fr, language='fr')
    y_pred = np.round(main.model_lstm_attention_fr(texts), 2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataOpinion(id=item.id, opinion=round(y_pred[i,0],2))
        responses.append(output_data)

    return responses