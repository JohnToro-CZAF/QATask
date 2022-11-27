from .serini_retriever import BM25Retriever
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
import re
import torch

class MiniLMBM25Retriever(BM25Retriever):
    def __init__(self, cfg, db_path) -> None:
        super().__init__(cfg.index_path, cfg.top_k, db_path)
        self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        self.cfg = cfg

    def preprocess(self, text):
      text = text.lower()
      text = text.replace("\n","")
      text = re.sub(' +', ' ', text)
      return text

    def __call__(self, data):
        data = super().__call__(data)
        for question in data:
            question_embedding = self.sentence_model.encode([self.preprocess(question['question'])], convert_to_tensor=True)
            contexts = [self.preprocess(passage[3]) for passage in question['candidate_passages']]
            tokenized_contexts = [sent_tokenize(context) for context in contexts]
            context_scores = []
            for context in tokenized_contexts:
                context_embedding = self.sentence_model.encode(context, convert_to_tensor=True)
                scores = torch.matmul(question_embedding, context_embedding.transpose(0, 1))
                scores = scores.mean().item()
                context_scores.append(scores)
            context_id_top_h = torch.argsort(torch.tensor(context_scores), descending=True)[:self.cfg.top_h]
            question['candidate_passages'] = [question['candidate_passages'][i] for i in context_id_top_h]
        print("Done")
        return data 




                
