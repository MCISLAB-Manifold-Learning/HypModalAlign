mkdir -p ./reproduce_logs/promptsrc
nshot=16

CUDA_VISIBLE_DEVICES='7' python train.py --config ./configs/few_shot/${nshot}-shot/imagenet/promptsrc+ours.yml --trial 1 > ./reproduce_logs/promptsrc/imagenet_${nshot}shot_train.txt 2>&1
                                                