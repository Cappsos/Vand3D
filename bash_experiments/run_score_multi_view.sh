
#python test.py \
 #   --config configs/infer_3d_ensemble_fuse.yaml \
#    --save_path ./test_results_dice3d/3D_fused/full_shot \
 #   --checkpoint_path ./experiments_3D/3D/full_shot_multi_scale/epoch_10.pth \
 #   --dice_threshold ./test_results_dice3d/3D_fused/full_shot/ \
#

python test.py \
    --config configs/infer_3d_ensemble_fuse.yaml \
   --save_path ./test_results_dice3d/3D_fused/full_shot \
   --checkpoint_path ./experiments_3D/3D/full_shot_multi_scale/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D_fused/full_shot/ \


python score.py \
    --config configs/score_3d_fuse.yaml \
    --results_root ./test_results_dice3d/3D_fused/full_shot \
    --threshold_json ./test_results_dice3d/3D_fused/full_shot/best_threshold.json \
    --volume_output_dir ./result_nibabel/3D_fused/full_shot \



# k=5 

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --checkpoint_path ./experiments_3D/3D/5_patient_rationale/epoch_15.pth \
    --save_path /mnt/external/results_nicoc/3D_final_final/k=5_ensemble \

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path /mnt/external/results_nicoc/3D_final_final/k=5_ensemble \
    --checkpoint_path ./experiments_3D/3D/5_patient_rationale/epoch_15.pth \
    --dice_threshold /mnt/external/results_nicoc/3D_final_final/k=5_ensemble \

python score_final.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root /mnt/external/results_nicoc/3D_final_final/k=5_ensemble \
    --threshold_json /mnt/external/results_nicoc/3D_final_final/k=5_ensemble/best_threshold.json  \


# k=5 fusion
python validation.py \
    --config configs/eval_3d_fuse.yaml \
    --checkpoint_path ./experiments_3D/3D/5_patient_multi_scale/epoch_14.pth \
    --save_path /mnt/external/results_nicoc/3D_final_final/k=5_fusion \

python test.py \
    --config configs/infer_3d_fuse.yaml \
    --save_path /mnt/external/results_nicoc/3D_final_final/k=5_fusion \
    --checkpoint_path ./experiments_3D/3D/5_patient_multi_scale/epoch_14.pth \
    --dice_threshold /mnt/external/results_nicoc/3D_final_final/k=5_fusion \

python score_final.py \
    --config configs/score_3d_fuse.yaml \
    --results_root /mnt/external/results_nicoc/3D_final_final/k=5_fusion \
    --threshold_json /mnt/external/results_nicoc/3D_final_final/k=5_fusion/best_threshold.json
    