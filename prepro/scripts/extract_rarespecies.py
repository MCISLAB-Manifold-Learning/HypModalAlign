import os
import re
import shutil
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
from argparse import ArgumentParser
from typing import Dict, List, Any, Tuple, Set

DEFAULT_ROOT_DIR = '/your_path_to_raw_rare_species'
DEFAULT_OUTPUT_DIR = '/your_path_to_extracted_rare_species'

def sanitize_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', name).lower()

def generate_unique_path(item: Dict[str, Any], path_set: Set[Tuple]) -> str:
    """
    wasted for now
    """

    full_path = (
        item['kingdom'], item['phylum'], item['class'],
        item['order'], item['family'], item['genus'], item['species']
    )
    
    sanitized_full = tuple(sanitize_name(x) for x in full_path)
    
    # try to use a shortest path that are different from other paths.
    for depth in range(1, 8):
        partial_path = sanitized_full[7-depth:]    
        if all(path[7-depth:] != partial_path for path in path_set if path != full_path):
            return '_'.join(partial_path)
    return '_'.join(sanitized_full)

def save_image(image_obj: Any, target_path: str) -> None:
    try:
        if isinstance(image_obj, Image.Image):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            image_obj.save(target_path)
        elif isinstance(image_obj, str) and os.path.exists(image_obj):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(image_obj, target_path)
        else:
            print(f"Illegal image: {type(image_obj)}")
    except Exception as e:
        print(f"Error when saving image: {e}")
        print(f"Save path: {target_path}")
        print(f"Error Info: {str(e)}")

def save_dataset_images(dataset: List[Dict[str, Any]], output_dir: str) -> None:
    """将数据集中的图片按分类保存到指定目录。"""
    keys = ['phylum', 'class', 'order', 'family', 'genus', 'species']
    for i, item in enumerate(dataset):
        # species = item['species'].lower()
        # unique_path = generate_unique_path(item, species_paths[species])
        path = '_'.join([item[k].lower() for k in keys])    
        
        filename = f"{item['rarespecies_id']}.jpg"
        output_path = os.path.join(output_dir, path, filename)
    
        try:
            save_image(item['file_name'], output_path)
        except Exception as e:
            print(f"Error when saving image: {str(e)}")
            print(f"Saving to: {output_path}")
    
            simple_path = os.path.join(output_dir, "error_backup", filename)
            os.makedirs(os.path.dirname(simple_path), exist_ok=True)
            try:
                save_image(item['file_name'], simple_path)
                print(f"saing to backup: {simple_path}")
            except:
                print("backup failed.")
        
        # 进度更新
        if (i + 1) % 100 == 0:
            print(f"processed {i + 1}/{len(dataset)} images | current tasks: {path}")

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=DEFAULT_ROOT_DIR)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    
    # 打印使用的路径
    print(f"rootdir: {args.root_dir}")
    print(f"outputdir: {args.output_dir}")
    
    print("loading dataset...")
    try:
        dataset = load_dataset(args.root_dir)['train']
        print(f"loading completed, {len(dataset)} samples in total")
    except Exception as e:
        print(f"loading failed: {e}")
        return
    
    print("saving...")
    save_dataset_images(dataset, args.output_dir)
    print(f"all images are saved to {args.output_dir}")

if __name__ == "__main__":
    main()