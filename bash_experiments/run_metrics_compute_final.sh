#python validation.py \
#   --config configs/eval_3d_ensemble.yaml \

#python test.py \
#    --config configs/infer_3d_ensemble.yaml \
  
#python score_final.py \
#    --config configs/score_3d_ensemble.yaml \


#python validation.py \
#   --config configs/eval_3d_fuse.yaml \

#python test.py \
#    --config configs/infer_3d_fuse.yaml \
  
#python score_final.py \
#    --config configs/score_3d_fuse.yaml \
  

python validation.py \
   --config configs/eval_2d.yaml \

python test.py \
    --config configs/infer_2d.yaml \

python score_final.py \
    --config configs/score_2d.yaml \

# 3D full shot ensamble
python validation.py \
   --config configs/eval_3d_ensemble.yaml \

python test.py \
    --config configs/infer_3d_ensemble.yaml \

python score_final.py \
    --config configs/score_3d_ensemble.yaml \


# 3D full shot no ensable

python validation.py \
   --config configs/eval_3d_no_ensemble.yaml \

python test.py \
    --config configs/infer_3d_no_ensemble.yaml \

python score_final.py \
    --config configs/score_3d_no_ensemble.yaml \


#3D fuse

python validation.py \
   --config configs/eval_3d_fuse.yaml \

python test.py \
    --config configs/infer_3d_fuse.yaml \

python score_final.py \
    --config configs/score_3d_fuse.yaml 


# MVFAAD

  

python score_final.py \
    --config configs/score_mvfaad.yaml 

