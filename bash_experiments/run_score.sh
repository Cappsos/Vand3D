python validation.py \
   --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/5_patient \
    --checkpoint_path ./experiments_3D/3D/5_patient_rationale/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/5_patient \
    --checkpoint_path ./experiments_3D/3D/5_patient_rationale/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/5_patient/ \
   


python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/5_patient \
    --threshold_json ./test_results_dice3d/3D/5_patient/best_threshold.json \





echo "Scoring for 5_patient done"

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/10_patient \
    --checkpoint_path ./experiments_3D/3D/10_patient_rationale/epoch_10.pth


python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/10_patient \
    --checkpoint_path ./experiments_3D/3D/10_patient_rationale/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/10_patient/ \



python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/10_patient \
    --threshold_json ./test_results_dice3d/3D/10_patient/best_threshold.json \


echo "Scoring for 10_patient done"

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/15_patient \
    --checkpoint_path ./experiments_3D/3D/15_patient_rationale/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/15_patient \
    --checkpoint_path ./experiments_3D/3D/15_patient_rationale/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/15_patient/ \



python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/15_patient \
    --threshold_json ./test_results_dice3d/3D/15_patient/best_threshold.json \

echo "Scoring for 15_patient done"

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/20_patient \
    --checkpoint_path ./experiments_3D/3D/20_patient/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/20_patient \
    --checkpoint_path ./experiments_3D/3D/20_patient/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/20_patient/ \


python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/20_patient \
    --threshold_json ./test_results_dice3d/3D/20_patient/best_threshold.json \

echo "Scoring for 20_patient done"


python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/30_patient \
    --checkpoint_path ./experiments_3D/3D/30_patient/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/30_patient \
    --checkpoint_path ./experiments_3D/3D/30_patient/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/30_patient/ \



python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/30_patient \
    --threshold_json ./test_results_dice3d/3D/30_patient/best_threshold.json \

echo "Scoring for 30_patient done"


python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/40_patient \
    --checkpoint_path ./experiments_3D/3D/40_patient/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/40_patient \
    --checkpoint_path ./experiments_3D/3D/40_patient/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/40_patient/ \



python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/40_patient \
    --threshold_json ./test_results_dice3d/3D/40_patient/best_threshold.json \

echo "Scoring for 40_patient done"

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/50_patient \
    --checkpoint_path ./experiments_3D/3D/50_patient/epoch_10.pth

python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/50_patient \
    --checkpoint_path ./experiments_3D/3D/50_patient/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/50_patient/ \


python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/50_patient \
    --threshold_json ./test_results_dice3d/3D/50_patient/best_threshold.json \

echo "Scoring for 50_patient done"

python validation.py \
    --config configs/eval_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/full_shot \
    --checkpoint_path ./experiments_3D/3D/full_shot/epoch_10.pth


python test.py \
    --config configs/infer_3d_ensemble.yaml \
    --save_path ./test_results_dice3d/3D/full_shot \
    --checkpoint_path ./experiments_3D/3D/full_shot/epoch_10.pth \
    --dice_threshold ./test_results_dice3d/3D/full_shot/ \



python score.py \
    --config configs/score_3d_ensemble.yaml \
    --results_root ./test_results_dice3d/3D/full_shot \
    --threshold_json ./test_results_dice3d/3D/full_shot/best_threshold.json \

echo "Scoring for full_shot done"