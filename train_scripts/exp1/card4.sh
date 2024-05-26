# 杂项：
# 学习率
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc lr_0.001 --lr 0.001 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/lr_0.001.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc lr_0.001 --lr 0.001 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/lr_0.001.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc lr_0.001 --lr 0.001 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/lr_0.001.out

# batch_size
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc bs_64 --batch_size 64 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/bs_64.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc bs_64 --batch_size 64 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/bs_64.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc bs_64 --batch_size 64 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/bs_64.out

# 对齐loss
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc relatedness_clip --class_feature_type clip --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/relatedness_clip.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc relatedness_clip --class_feature_type clip --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/relatedness_clip.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc relatedness_clip --class_feature_type clip --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/relatedness_clip.out

CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc relatedness_w2v --class_feature_type w2v --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/relatedness_w2v.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc relatedness_w2v --class_feature_type w2v --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/relatedness_w2v.out
CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc relatedness_w2v --class_feature_type w2v --relatedness_loss_coef 1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/relatedness_w2v.out



