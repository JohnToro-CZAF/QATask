```
python3 -m tools.finetune_reader.electra.train \
  --model_name_or_path 'FPTAI/velectra-base-discriminator-cased' \
  --do_eval \
  --do_train \
  --per_gpu_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/electra' \
  --overwrite_output_dir \
  --save_steps 1000 \ 
  -n_gpu 4
```