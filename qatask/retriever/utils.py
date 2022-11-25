import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import enum

from underthesea import word_tokenize, ner, pos_tag

from underthesea import word_tokenize, text_normalize, sent_tokenize
from underthesea import word_tokenize, text_normalize, sent_tokenize
from rank_bm25 import BM25Okapi

def nlp_sieve_passages(question, passages, k):
    question = word_tokenize(text_normalize(question))
    tokenized_passages = []
    for passage in passages:
        sentences = sent_tokenize(passage)
        cleaned_sentences = [text_normalize(sent) for sent in sentences]
        cleaned_sentences = [sent for sent in sentences]
        tokenized_passage = []
        for sent in cleaned_sentences:
            tokenized_passage.extend(word_tokenize(sent))
        tokenized_passages.append(tokenized_passage)
    bm25 = BM25Okapi(tokenized_passages)
    topk_passages = bm25.get_top_n(question, passages, n=k)
    scores = bm25.get_scores(question)
    return topk_passages, scores

class QPM(enum.Enum):
  Unclassified = 0
  SimpleFact = 1
  ComplexFact = 2

class QuestionEntity(enum.Enum):
  NONE = 1
  PERSON = 2
  LOCATION = 3
  ORGANIZATION = 4

# class QPM(object):
#     """
#         QPM pre-processes the original question implementing a basic gramatical correction, and sanitizing the question to
#         remove junk data.
#     """
#     def __init__(self, question):
#         self.question = question
#         self.qpm = QPM.Unclassified
#         self.question_entity = QuestionEntity.NONE
#         self.question = self.__preprocess_question()
#         self.question = self.__sanitize_question()

#     def __preprocess_question(self):
#         """
#             Pre-process the question to correct the gramatical errors.
#         """
#         return self.question
#     def __santiize_question(self):
#         """
#             Sanitize the question to remove junk data.
#         """
#         return self.question
#     def get_question(self):
#         """
#             Returns the question.
#         """
#         return self.question
#     def 