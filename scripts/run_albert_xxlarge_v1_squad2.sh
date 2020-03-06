export SQUAD_DIR="data/squad-v2.0"


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 run_squad.py \
    --model_type albert \
    --model_name_or_path "/users5/dkhao/data/transformers/albert-xxlarge-v1" \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file $SQUAD_DIR/train-v2.0.json \
    --predict_file $SQUAD_DIR/dev-v2.0.json \
    --version_2_with_negative \
    --max_steps 8144 \
    --warmup_steps 814 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir outputs/albert_xxlarge_v1_squad2/ \
    --per_gpu_eval_batch_size=48   \
    --per_gpu_train_batch_size=1   \
    --gradient_accumulation_steps 24 \
    --save_steps 2000 \