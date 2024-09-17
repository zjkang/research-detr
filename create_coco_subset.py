import argparse
import os
import random
import shutil
import json
from pycocotools.coco import COCO

def create_coco_subset(coco_path, output_path, num_images):
    # Load COCO dataset
    coco = COCO(coco_path)

    # Determine if we are working with the train or val set
    set_name = 'train2017' if 'train' in os.path.basename(coco_path) else 'val2017'

    # Get all image ids
    img_ids = coco.getImgIds()

    # Randomly select a subset of image ids
    subset_img_ids = random.sample(img_ids, num_images)

    # Create new directories for subset
    images_dir = os.path.join(output_path, 'images', set_name)
    annotations_dir = os.path.join(output_path, 'annotations')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Copy the selected images
    for img_id in subset_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        src_img_path = os.path.join(os.path.dirname(coco_path).replace('annotations', 'images'), set_name, img_info['file_name'])
        
        # Print the source image path for debugging
        print(f"Copying {src_img_path} to {images_dir}")

        dst_img_path = os.path.join(images_dir, img_info['file_name'])
        
        # Check if the source file exists
        if not os.path.exists(src_img_path):
            print(f"File not found: {src_img_path}")
            continue
        
        shutil.copy(src_img_path, dst_img_path)

    # Save the new annotations
    small_coco = {
        "images": [coco.loadImgs(img_id)[0] for img_id in subset_img_ids],
        "annotations": coco.loadAnns(coco.getAnnIds(imgIds=subset_img_ids)),
        "categories": coco.loadCats(coco.getCatIds())
    }
    
    with open(os.path.join(annotations_dir, os.path.basename(coco_path)), 'w') as f:
        json.dump(small_coco, f)
    
    print(f"Subset created with {num_images} images and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a small COCO subset")
    parser.add_argument('--coco_path', type=str, required=True, help="Path to the COCO annotations file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the small subset")
    parser.add_argument('--num_images', type=int, required=True, help="Number of images for the subset")

    args = parser.parse_args()

    create_coco_subset(args.coco_path, args.output_path, args.num_images)

