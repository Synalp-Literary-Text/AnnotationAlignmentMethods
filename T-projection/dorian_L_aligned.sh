accelerate launch --mixed_precision bf16 --num_processes 1 --num_machines 1 seq2seq_t5.py \
  --train_tsv Dorian_al/en.aligned_en_dorian.REFQUOTE_full_2.conll \
  --dev_tsv Dorian_al/en.aligned_en_dorian.REFQUOTE_3_2.conll \
  --test_source_tsv Dorian_al/en.aligned_en_dorian.REFQUOTE_3_2.conll \
  --test_target_tsv Dorian_al/fr.dorian_aligned.REFQUOTE_3_2.conll \
  --num_beams 15 \
  --num_return_sequences 15 \
  --model_name_or_path google/mt5-xl \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --per_device_test_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_train_epochs 6 \
  --output_dir Dorian_al/mt5-large \
  --seed 42 \
  --eval_every 2 \
  --max_source_length 256 \
  --max_target_length 256 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 500 \
  --project_name "T-Projection-Dorian"
  
  # 2) Compute translation probabilities for each candidate

python3 calculate_scores_nmts.py \
  --jsonl_path Dorian_al/mt5-large/fr.dorian_aligned.REFQUOTE_3_2.jsonl \
  --model_name_or_path facebook/m2m100_418M \
  --output_path Dorian_al/mt5-large/fr.dorian_aligned.REFQUOTE_3_2.json \
  --source_lang en \
  --target_lang fr \
  --normalize \
  --both_directions

# 3) Label projection
 
python3 label_projection.py \
  --jsonl_path Dorian_al/mt5-large/fr.dorian_aligned.REFQUOTE_3_2.jsonl \
  --dictionary_path Dorian_al/mt5-large/fr.dorian_aligned.REFQUOTE_3_2.json \
  --output_path Dorian_al/mt5-large/en2fr.dorian_aligned.REFQUOTE_3_2.conll
