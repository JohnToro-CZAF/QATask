```
python3 -m tools.finetune_reader.xlm_r.train \
  --model_name_or_path xlm-roberta-large \
  --do_eval \
  --do_train \
  --per_device_train_batch_size 5 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --max_seq_length 256 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/xlm-roberta-large' \
  --overwrite_output_dir \
  --version_2_with_negative \
  --save_steps 1000 \ 
```
Eval
```
python3 -m tools.finetune_reader.xlm_r.train \
  --model_name_or_path 'checkpoint/pretrained_model/electra/checkpoint-19000' \
  --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/electra' 
  --version_2_with_negative
```