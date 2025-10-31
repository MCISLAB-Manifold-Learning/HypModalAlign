mkdir -p ./reproduce_logs/maple
nshot=1

CUDA_VISIBLE_DEVICES='0' python train.py --config ./configs/few_shot/${nshot}-shot/cifar100/maple+ours.yml --trial 1 > ./reproduce_logs/maple/cifar100_${nshot}shot_train.txt 2>&1
                                                