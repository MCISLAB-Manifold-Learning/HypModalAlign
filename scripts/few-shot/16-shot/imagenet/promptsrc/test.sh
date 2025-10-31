batch_size=512 #ajust based on your GPU memory budget
nshot=16
mkdir -p ./reproduce_logs/promptsrc/

# evaluate with the help of a clip pretrained by vanilla prompt tuning
# pretrained_pt_clip_dir="pretrained_pt_clip_dir"
# evaluate LA, HCA
# CUDA_VISIBLE_DEVICES='0' python reeval.py --folder ./runs/imagenet/promptsrc/ViT-B_16/few_shot/${nshot}-shot/ours/trial_1 --bz ${batch_size} --subsample all --extra_pt_clip_ks '[5, 10]' --pretrained_pt_clip_dir $pretrained_pt_clip_dir > ./reproduce_logs/promptsrc/imagenet_${nshot}shot_reeval.txt  2>&1
                
# evlauate MTA               
CUDA_VISIBLE_DEVICES='0' python evalmta.py --folder ./runs/imagenet/promptsrc/ViT-B_16/few_shot/${nshot}-shot/ours/trial_1 --bz ${batch_size} --subsample all > ./reproduce_logs/promptsrc/imagenet_${nshot}shot_evalmta.txt 2>&1
                