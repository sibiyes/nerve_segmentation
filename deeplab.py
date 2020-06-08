import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

from deeplab.model import Deeplabv3

def main():
    print(Deeplabv3)
    deeplab_model = Deeplabv3(input_shape=(None, None, 3), classes=4)
    
if __name__ == '__main__':
    main()
