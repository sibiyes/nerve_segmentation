import sys
import os
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

def main():
    data_dir = base_dir + '/ultrasound-nerve-segmentation/train'
    data_output_dir = base_dir + '/ultrasound-nerve-segmentation/train_jpeg'
    
    image_paths = [x for x in os.listdir(data_dir) if x.endswith('.tif')]
    
    for img_f in image_paths:
        img = Image.open(os.path.join(data_dir, img_f))
        print(img)
        img.save(os.path.join(data_output_dir, img_f.replace('.tif', '.jpeg')))
        
        
if __name__ == '__main__':
    main()
