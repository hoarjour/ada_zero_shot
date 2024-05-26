# SUN:
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc benchmark --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/benchmark.out

# 测试num_query
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc num_queries_200 --num_queries 200 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/num_queries_200.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc num_queries_500 --num_queries 500 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/num_queries_500.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc num_queries_50 --num_queries 50 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/num_queries_50.out

# 测试分割的参数
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_15_c_1e-3 --num_queries 100 --markers 15 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_15_c_1e-3.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_25_c_1e-3 --num_queries 100 --markers 25 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_25_c_1e-3.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_50_c_1e-3 --num_queries 100 --markers 50 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_50_c_1e-3.out

CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_9_c_0 --num_queries 100 --markers 9 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_9_c_0.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_15_c_0 --num_queries 100 --markers 15 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_15_c_0.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_25_c_0 --num_queries 100 --markers 25 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_25_c_0.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc m_50_c_0 --num_queries 100 --markers 50 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/m_50_c_0.out

# 测试loss平衡
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc box_3_giou_1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 3 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/box_3_giou_1.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc box_3_giou_2 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 3 --giou_loss_coef 2 --enc_layers 2 --dec_layers 2 >> logs/SUN/box_3_giou_2.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc class_4 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 4 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/class_4.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc penalty_2 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 2 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/SUN/penalty_2.out

# 测试层数
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc layer_1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 1 --dec_layers 1 >> logs/SUN/layer_1.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset SUN --desc layer_3 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 3 --dec_layers 3 >> logs/SUN/layer_3.out
