batch_size=512 #ajust based on your GPU memory budget
nshot=16
mkdir -p ./reproduce_logs/promptsrc/

#Optional. Because the train.py will automatically test acc and consistency after training.
# CUDA_VISIBLE_DEVICES='0' python reeval.py --folder ./runs/cifar100/promptsrc/ViT-B_16/few_shot/${nshot}-shot/ours/trial_1 --bz ${batch_size} --subsample all > ./reproduce_logs/promptsrc/cifar100_${nshot}shot_reeval.txt  2>&1
                
CUDA_VISIBLE_DEVICES='0' python evalmta.py --folder ./runs/cifar100/promptsrc/ViT-B_16/few_shot/${nshot}-shot/ours/trial_1 --bz ${batch_size} --subsample all > ./reproduce_logs/promptsrc/cifar100_${nshot}shot_evalmta.txt 2>&1
                