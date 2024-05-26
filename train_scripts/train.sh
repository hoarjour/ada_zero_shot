CUDA_VISIBLE_DEVICES=3 python learn_traindata_semantics.py --not_local --class_feature_type clip --relatedness_loss_coef 1 --epochs 1 --use_pretrained_features --dataset CUB --desc benchmark --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/benchmark.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --class_feature_type clip --relatedness_loss_coef 1 --epochs 1 --use_pretrained_features --dataset SUN --desc benchmark --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/benchmark.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --class_feature_type clip --relatedness_loss_coef 1 --epochs 1 --use_pretrained_features --dataset AWA2 --desc benchmark --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/benchmark.out
python temp.py --p 0 >> logs/0.out

python temp.py --p 1 >> logs/1.out
