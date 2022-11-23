import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import enum

from underthesea import word_tokenize, ner, pos_tag

from underthesea import word_tokenize, text_normalize, sent_tokenize
from rank_bm25 import BM25L

def nlp_sieve_passages(passages, question):
    tokenized_passages = []
    question = word_tokenize(text_normalize(question), format="text")
    poss = pos_tag(question)
    # print(poss)
    query = " ".join([w for w, pos in poss if pos != 'E' and pos != 'P' and pos != 'V' and pos != 'A'])
    print(query)
    for id, wiki, score, passage in passages:
        tokenized_passages.append(word_tokenize(text_normalize(passage), format="text"))
    bm25 = BM25L(tokenized_passages)
    top5_passages = bm25.get_top_n(question, tokenized_passages, n=5)
    formated_passages = []
    for passage in top5_passages:
        print(passage)
        formated_passages.append(passages[tokenized_passages.index(passage)])
    # print(formated_passages)
    return formated_passages

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