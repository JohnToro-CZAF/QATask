```
python3 -m tools.finetune_reader.T5.train \
  --model_name_or_path VietAI/vit5-base \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/T5' \
  --overwrite_output_dir \
  --version_2_with_negative \
  --gradient_checkpointing True
```
```
python3 -m tools.finetune_reader.T5.train \
  --model_name_or_path 'checkpoint/pretrained_model/T5/checkpoint-11500/' \
  --do_eval \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/T5' \
  --version_2_with_negative \
  --predict_with_generate
```
```
python3 -m tools.finetune_reader.T5.train \
  --model_name_or_path google/flan-t5 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 10 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir 'checkpoint/pretrained_model/T5' \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --version_2_with_negative \
  --gradient_checkpointing True
```