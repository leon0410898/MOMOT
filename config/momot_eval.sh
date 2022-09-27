# for MOT17

EXP_DIR=exps/momot
python3 eval.py \
     --meta_arch momot \
     --dataset_file e2e_mot \
     --mot_path /datasets \
     --epoch 100 \
     --lr_drop 100 \
     --lr 2e-4 \
     --eval \
     --lr_backbone 2e-5 \
     --output_dir ${EXP_DIR} \
     --batch_size 1 \
     --sample_mode 'random_interval' \
     --sample_interval 10 \
     --sampler_steps 50 90 120 \
     --sampler_lengths 2 3 4 5 \
     --dropout 0.1 \
     --no_aux_loss \
     --num_workers 1 \
     --data_txt_path_train datasets/data_path/mot17.train \
     --data_txt_path_val datasets/data_path/mot17.train \
     --resume ${EXP_DIR}/momot_final.pth \
     --visibility_thresh 0.0 \
     --debug