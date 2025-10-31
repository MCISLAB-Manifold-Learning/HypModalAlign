mkdir -p ./reproduce_logs/maple
nshot=1

CUDA_VISIBLE_DEVICES='4' python train.py --config ./configs/few_shot/${nshot}-shot/rarespecies/maple+ours.yml --trial 1 > ./reproduce_logs/maple/rarespecies_${nshot}shot_train.txt 2>&1
                                                