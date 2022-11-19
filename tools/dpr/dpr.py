import logging 
import os

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

import haystack
from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

# Train DPR on custom dataset
def main():
  pwd = os.getcwd()
  doc_dir = pwd+'/datasets/data_dpr/'
  train_filename = doc_dir + 'dpr_train.json'
  dev_filename = doc_dir + 'dpr_dev.json'
  test_filename = doc_dir + 'dpr_test.json'
  query_model = "trituenhantaoio/bert-base-vietnamese-uncased"
  passage_model = "trituenhantaoio/bert-base-vietnamese-uncased"

  save_dir = "checkpoint/dpr_zalo_v3/"

  retriever = DensePassageRetriever(document_store=InMemoryDocumentStore(),
                                    query_embedding_model=query_model,
                                    passage_embedding_model=passage_model,
                                    max_seq_len_query=64,
                                    max_seq_len_passage=256,
                                    )
  retriever.train(
      data_dir=doc_dir,
      train_filename=train_filename,
      dev_filename=dev_filename,
      test_filename=test_filename,
      n_epochs=2,
      batch_size=32,
      grad_acc_steps=8,
      save_dir=save_dir,
      evaluate_every=1000,
      embed_title=True,
      num_positives=1,
      num_hard_negatives=1,
  )

if __name__ == '__main__':
  main()