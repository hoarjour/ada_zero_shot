# CUB
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc benchmark --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/benchmark.out

# 测试num_query
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc num_queries_200 --num_queries 200 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/num_queries_200.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc num_queries_500 --num_queries 500 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/num_queries_500.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc num_queries_50 --num_queries 50 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/num_queries_50.out

# 测试分割的参数
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc m_15_c_1e-3 --num_queries 100 --markers 15 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/m_15_c_1e-3.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc m_25_c_1e-3 --num_queries 100 --markers 25 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/m_25_c_1e-3.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc m_50_c_1e-3 --num_queries 100 --markers 50 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/m_50_c_1e-3.out

CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc m_9_c_0 --num_queries 100 --markers 9 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/m_9_c_0.out
CUDA_VISIBLE_DEVICES=2 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset CUB --desc m_15_c_0 --num_queries 100 --markers 15 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/CUB/m_15_c_0.out
