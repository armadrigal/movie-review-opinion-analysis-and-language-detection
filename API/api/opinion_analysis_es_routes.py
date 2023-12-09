from fastapi import APIRouter
from typing import List
from .data_models import InputData
from .opinion_analysis_es import predict

router = APIRouter()

@router.post("/OpinionAnalysis/es/lstm", summary="Opinion Analysis, Language: Spanish, Model: LSTM")
async def opinion_es_lstm(data: List[InputData]):
    responses = predict(data, model="lstm")
    return responses

@router.post("/OpinionAnalysis/es/gru", summary="Opinion Analysis, Language: Spanish, Model: GRU")
async def opinion_es_gru(data: List[InputData]):
    responses = predict(data, model="gru")
    return responses

@router.post("/OpinionAnalysis/es/cnn", summary="Opinion Analysis, Language: Spanish, Model: CNN")
async def opinion_es_gru(data: List[InputData]):
    responses = predict(data, model="cnn")
    return responses

@router.post("/OpinionAnalysis/es/attention", summary="Opinion Analysis, Language: Spanish, Model: Attention")
async def opinion_es_gru(data: List[InputData]):
    responses = predict(data, model="attention")
    return responses

@router.post("/OpinionAnalysis/es/lstm+attention", summary="Opinion Analysis, Language: Spanish, Model: GRU")
async def opinion_es_gru(data: List[InputData]):
    responses = predict(data, model="lstm+attention")
    return responses