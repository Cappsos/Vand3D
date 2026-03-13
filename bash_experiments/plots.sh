python plots_thesis_volumes.py \
 --pred_root /mnt/external/results_nicoc/3D_final_final/full_shot_fused \
 --gt_root /mnt/external/data/BraTS_3D \
 --out_root /mnt/external/results_nicoc/plots_volumes/full_shot_fused \
 --reconstruct_mode subvolume \
 --prefix anomaly_map_depth_ \
 --depth 32 \
 --full_depth 155 \
 --target_h 240 \
 --target_w 240 \
 --threshold 1.02 \
 --patients "" \
 --zrev \
 --zshift 0 \

python plots_thesis_volumes.py \
 --pred_root /mnt/external/results_nicoc/3D_final_final/full_shot_2d_replicated \
 --gt_root /mnt/external/data/BraTS_3D \
 --out_root /mnt/external/results_nicoc/plots_volumes/full_shot_2d_replicated \
 --reconstruct_mode slice \
 --prefix anomaly_map_depth_ \
 --depth 32 \
 --full_depth 155 \
 --target_h 240 \
 --target_w 240 \
 --threshold 1.32 \
 --patients "" \
 --zrev \
 --zshift 0 \



python plots_thesis_volumes.py \
 --pred_root /mnt/external/results_nicoc/3D_final_final/k=5_ensemble \
 --gt_root /mnt/external/data/BraTS_3D \
 --out_root /mnt/external/results_nicoc/plots_volumes/k=5_ensemble \
 --reconstruct_mode subvolume \
 --prefix anomaly_map_depth_ \
 --depth 32 \
 --full_depth 155 \
 --target_h 240 \
 --target_w 240 \
 --threshold 1.38 \
 --patients "" \
 --zrev \
 --zshift 0 \

python plots_thesis_volumes.py \
 --pred_root /mnt/external/results_nicoc/3D_final_final/k=5_fusion \
 --gt_root /mnt/external/data/BraTS_3D \
 --out_root /mnt/external/results_nicoc/plots_volumes/k=5_fusion \
 --reconstruct_mode subvolume \
 --prefix anomaly_map_depth_ \
 --depth 32 \
 --full_depth 155 \
 --target_h 240 \
 --target_w 240 \
 --threshold 1.38 \
 --patients "" \
 --zrev \
 --zshift 0