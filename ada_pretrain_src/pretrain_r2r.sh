NODE_RANK=0
NUM_GPUS=2

CUDA_VISIBLE_DEVICES='0,1' python ada_pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --output_dir datasets/R2R/exprs/pretrain/test \
    --model_config ada_pretrain_src/config/r2r_model_config.json \
    --config ada_pretrain_src/config/pretrain_r2r.json \
    --checkpoint datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt