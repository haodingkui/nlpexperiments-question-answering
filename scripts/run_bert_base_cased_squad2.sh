export SQUAD_DIR="data/squad-v2.0"


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 run_squad.py \
    --model_type bert \
    --model_name_or_path "/users5/dkhao/data/transformers/bert-base-cased-squad2" \
    --do_eval \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --version_2_with_negative \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir outputs/bert_base_cased_squad2/ \
    --per_gpu_eval_batch_size=48   \
    --per_gpu_train_batch_size=1   \
    --gradient_accumulation_steps 24 \
    --save_steps 2000 \