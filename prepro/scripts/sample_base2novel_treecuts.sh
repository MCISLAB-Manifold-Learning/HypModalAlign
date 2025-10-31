#!/bin/bash
cd ../..  # 切换到项目根目录

# 定义数据集和子样本数组（使用小写变量名符合规范）
datasets=("cifar100" "imagenet" "sun" "rarespecies")
subsamples=("base" "novel")

seed=100  # 与cfg中的seed相同
ntree=25

# 创建目标目录（确保存在）
mkdir -p "./prepro/data"  # 递归创建目录[7](@ref)

# 嵌套循环处理每个数据集和子样本
for dataset in "${datasets[@]}"; do  # 正确引用数组[3,5](@ref)
    for subsample in "${subsamples[@]}"; do
        # 输出详细执行信息
        echo "🚀 正在处理数据集: ${dataset}, 子样本: ${subsample}"
        echo "  执行 sample_treecuts.py..."
        
        # 执行Python脚本
        python sample_treecuts.py \
            --config "./configs/${dataset}/treecut/sample.yml" \
            --subsample "${subsample}" \
            --multi
        
        # 源文件和目标路径
        src_file="./runs/${dataset}/treecuts/multi/seed${seed}/treecuts_${ntree}_multi_${subsample}.pkl"
        dest_dir="./prepro/data/${dataset}"
        
        # # 创建目标目录（确保存在）
        # mkdir -p "${dest_dir}"  # 防止目录不存在[7](@ref)
        
        # 复制文件并输出状态
        echo "  复制文件: ${src_file} → ${dest_dir}/"
        cp -v "${src_file}" "${dest_dir}/"  # -v显示复制详情
        
        echo "✅ 完成: ${dataset} - ${subsample}"
        echo "----------------------------------------"
    done
done
