CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc m_50_c_0 --num_queries 100 --markers 50 --compactness 0 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/m_50_c_0.out

# 测试loss平衡
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc box_3_giou_1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 3 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/box_3_giou_1.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc box_3_giou_2 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 3 --giou_loss_coef 2 --enc_layers 2 --dec_layers 2 >> logs/AWA2/box_3_giou_2.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc class_4 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 4 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/class_4.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc penalty_2 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 2 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/penalty_2.out

# 测试层数
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc layer_1 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 1 --dec_layers 1 >> logs/AWA2/layer_1.out
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc layer_3 --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 3 --dec_layers 3 >> logs/AWA2/layer_3.out

# 测试w2v
CUDA_VISIBLE_DEVICES=1 python learn_traindata_semantics.py --not_local --use_pretrained_features --dataset AWA2 --desc w2v_align --class_feature_type w2v --num_queries 100 --markers 9 --compactness 0.001 --class_loss_coef 3 --penalty_coef 1 --box_loss_coef 2 --giou_loss_coef 1 --enc_layers 2 --dec_layers 2 >> logs/AWA2/w2v_align.out
