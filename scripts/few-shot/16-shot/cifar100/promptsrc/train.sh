mkdir -p ./reproduce_logs/promptsrc
nshot=16

CUDA_VISIBLE_DEVICES='1' python train.py --config ./configs/few_shot/${nshot}-shot/cifar100/promptsrc+ours.yml --trial 1 > ./reproduce_logs/promptsrc/cifar100_${nshot}shot_train.txt 2>&1
                                                