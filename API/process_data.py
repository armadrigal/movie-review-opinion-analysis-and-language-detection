import re 

class ProcessData:

    def __new__(cls, texts, vocabulary, language=None):
        pass

    @classmethod
    def _cleaning_text(cls, text):
        text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ\s\'-]', ' ', text)
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    @classmethod
    def _cleaning_text_es(text):
        text = re.sub(r'[^a-zA-ZáéíóúñÁÉÍÓÚÑ]', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text

    @classmethod
    def _cleaning_text_en(text):
        text = re.sub(r'[^a-zA-Z\s\'-]', '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text