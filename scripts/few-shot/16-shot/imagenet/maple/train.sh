mkdir -p ./reproduce_logs/maple
nshot=16

CUDA_VISIBLE_DEVICES='6' python train.py --config ./configs/few_shot/${nshot}-shot/imagenet/maple+ours.yml --trial 1 > ./reproduce_logs/maple/imagenet_${nshot}shot_train.txt 2>&1
                                                