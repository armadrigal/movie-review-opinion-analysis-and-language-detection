from fastapi import APIRouter
from typing import List
from .data_models import InputData
from .opinion_analysis_en import predict

router = APIRouter()

@router.post("/OpinionAnalysis/en/lstm", summary="Opinion Analysis, Language: English, Model: LSTM")
async def opinion_en_lstm(data: List[InputData]):
    responses = predict(data, model="lstm")
    return responses

@router.post("/OpinionAnalysis/en/gru", summary="Opinion Analysis, Language: English, Model: GRU")
async def opinion_en_gru(data: List[InputData]):
    responses = predict(data, model="gru")
    return responses

@router.post("/OpinionAnalysis/en/cnn", summary="Opinion Analysis, Language: English, Model: CNN")
async def opinion_en_cnn(data: List[InputData]):
    responses = predict(data, model="cnn")
    return responses

@router.post("/OpinionAnalysis/en/attention", summary="Opinion Analysis, Language: English, Model: Attention")
async def opinion_en_attention(data: List[InputData]):
    responses = predict(data, model="attention")
    return responses

@router.post("/OpinionAnalysis/en/lstm+attention", summary="Opinion Analysis, Language: English, Model: LSTM+Attention")
async def opinion_en_lstm_attention(data: List[InputData]):
    responses = predict(data, model="lstm+attention")
    return responses