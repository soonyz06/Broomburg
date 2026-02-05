from pathlib import Path
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import nltk
#nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
import os

class SimpleBertChunker:
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased') #doesn't use bert-base-uncased's MLM head
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)
        self._model.eval() #eval mode -> all neurons active, how to download the models

    def _nsp_probability(self, sent1, sent2): #next sentence prediction -> prob
        encoding = self._tokenizer(sent1, sent2, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**encoding)
            prob = torch.softmax(outputs.logits, dim=1)[0][0].item() 
        return prob

    def chunk_text(self, text, threshold=0.5, min_buffer_size=2):
        sentences = sent_tokenize(text) #[sent1, ..., sentn]
        if len(sentences)<=1:
            return [text]

        chunks = []
        buffer = [sentences[0]]
        for i in range(1, len(sentences)):
            nsp_prob = self._nsp_probability(buffer[-1] , sentences[i]) 
            if nsp_prob<threshold and len(buffer) >= min_buffer_size:
                chunks.append(' '.join(buffer))
                buffer = [sentences[i]]
            else:
                buffer.append(sentences[i])

        if buffer:
            chunks.append(' '.join(buffer))
        return chunks

    def print_chunk(self, chunks):
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        return self


