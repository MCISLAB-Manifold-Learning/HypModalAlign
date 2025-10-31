mkdir -p ./reproduce_logs/maple
nshot=1
CUDA_VISIBLE_DEVICES='2' python train.py --config ./configs/few_shot/${nshot}-shot/sun/maple+ours.yml --trial 1 > ./reproduce_logs/maple/sun_${nshot}shot_train.txt 2>&1
                                                