from fastapi import APIRouter
from typing import List
from DataModels import InputData, OutputDataOpinion
from process_data import ProcessData
import main
import numpy as np

router = APIRouter()

@router.post("/OpinionAnalysis/it/gru")
async def opinion_it_gru(data: List[InputData]):

    texts = [item.text for item in data]
    texts = ProcessData(texts, main.vocabulary_it, main.stopwords_it, language='it')
    y_pred = np.round(main.model_gru_it(texts), 2)

    responses = []
    for i, item in enumerate(data):
        output_data = OutputDataOpinion(id=item.id, opinion=round(y_pred[i,0],2))
        responses.append(output_data)

    return responses