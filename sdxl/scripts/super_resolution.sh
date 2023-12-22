#!/bin/zsh

for image_file in /mnt/homes/minghao/AI/final_project/src/im_variations/samples/*.png
do
    echo "$image_file" 
    python /mnt/homes/minghao/AI/final_project/sdxl/scripts/superresolution.py --image=$image_file
done