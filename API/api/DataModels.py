from pydantic import BaseModel
from typing import List

class InputData(BaseModel):
    id: int
    text: str

class InputTexts(BaseModel):
    id: int
    text: str

class InputDataCustomLanguage(BaseModel):
    model: str
    texts: List[InputTexts]

class OutputDataOpinion(BaseModel):
    id: int
    opinion: float

class OutputDataLanguage(BaseModel):
    id: int
    es: float
    en: float
    fr: float
    pt: float
    it: float

class LanguagePrediction(BaseModel):
    es: float
    en: float
    fr: float
    pt: float
    it: float

class OutputDataAllModelsLanguage(BaseModel):
    id: int
    mlp: LanguagePrediction
    lstm: LanguagePrediction
    gru: LanguagePrediction
