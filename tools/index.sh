CUDA_VISIBLE_DEVICES=$7 python -m pyserini.encode \
  input   --corpus $1 \
          --fields text  \
          --delimiter "\n"  \
          --shard-id $2   \
          --shard-num $3 \
  output  --embeddings $4 \
          --to-faiss \
  encoder --encoder $5 \
          --fields text \
          --batch $6 \
          --fp16 