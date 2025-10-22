CUDA_VISIBLE_DEVICES=0 python evaluate.py \
-c ./config/isic_2017/sfinet.py \
-o ./fig_results_512/isic2017/sfinet/ --rgb -t "lr"  

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
-c ./config/isic_2018/sfinet.py \
-o ./fig_results_512/isic2018/sfinet/ --rgb -t "lr"  