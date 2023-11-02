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

    @classmethod
    def _tokenize_texts(cls, texts):
        tokenized_sentences = []
        for text in texts:
            tokenized_sentences.append(text.split())
        return tokenized_sentences

    @classmethod
    def _vectorice_texts(cls, texts, vocabulary):
        vectoriced_sentences = []
        for text in texts:
            vectoriced_sentence = []
            for word in text:
                if word in vocabulary:
                    vectoriced_sentence.append(vocabulary[word])
                else:
                    vectoriced_sentence.append(0)
            vectoriced_sentences.append(vectoriced_sentence)
        return vectoriced_sentences