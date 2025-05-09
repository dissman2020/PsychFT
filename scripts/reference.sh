accelerate launch src/reference.py \
--teacher_model "/data/kankan.lan/repos/Llama-3.1-Centaur-8B-adapter/Centaur-8B" \
--dataset_name "/data/kankan.lan/datasets/marcelbinz/Psych-101-all" \
--dataset_split "train" \
--batch_size 2 \
--max_sequence_length 8192 \
