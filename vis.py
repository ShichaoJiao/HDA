import argparse
import glob
import os
import shutil

import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from model import Generator
from utils import get_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vis Generator')
    parser.add_argument('--sketch_name', default='/media/jiao/datasets/SHREC13_ZS_DTO/sketch/train/shark', type=str,
                        help='Sketch image name')
    parser.add_argument('--generator_name', default='best/SHREC13_ZS_DTO_resnet50_512_generator.pth', type=str,
                        help='Generator name')
    parser.add_argument('--save_root', default='result', type=str, help='Result saved root path')

    opt = parser.parse_args()

    sketch_names, generator_name, save_root = opt.sketch_name, opt.generator_name, opt.save_root

    generator = Generator(in_channels=8, num_block=8)
    generator.load_state_dict(torch.load(generator_name, map_location='cpu'))
    generator = generator.cuda()
    generator.eval()

    sketch_names = glob.glob('{}/*.png'.format(sketch_names))

    for sketch_name in tqdm(sketch_names):
        sketch = get_transform('val')(Image.open(sketch_name).convert('RGB')).unsqueeze(dim=0).cuda()
        with torch.no_grad():
            photo = generator(sketch,is_target = 0)

        result_path = '{}/{}'.format(save_root, os.path.basename(sketch_name).split('.')[0])
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.mkdir(result_path)

        Image.open(sketch_name).resize((224, 224), resample=Image.BILINEAR).save('{}/sketch.png'.format(result_path))
        ToPILImage()((((photo.squeeze(dim=0) + 1.0) / 2) * 255).byte().cpu()).save('{}/photo.png'.format(result_path))
