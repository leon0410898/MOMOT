# for MOT17

EXP_DIR=exps/momot
python3 submit.py \
     --meta_arch momot \
     --dataset_file e2e_mot \
     --mot_path /datasets \
     --use_checkpoint \
     --epoch 100 \
     --lr_drop 75 \
     --lr 2e-4 \
     --lr_backbone 2e-5 \
     --output_dir ${EXP_DIR} \
     --batch_size 1 \
     --sample_mode 'random_interval' \
     --sample_interval 5 \
     --sampler_steps 20 40 60 80 \
     --sampler_lengths 3 4 5 6 7\
     --dropout 0.1 \
     --no_aux_loss \
     --num_workers 1 \
     --freeze_det \
     --losses pred_track_logits pred_track_boxes \
     --data_txt_path_train datasets/data_path/mot17.train \
     --data_txt_path_val datasets/data_path/mot17.train \
     --resume exps/momot/momot_final.pth \
     --exp_name pub_submit_17 \
     --visibility_thresh 0.0 \
     --iou_thresh -1 \
     --miss_tolerance 3 \
     --debug