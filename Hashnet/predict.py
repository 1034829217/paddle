#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
import argparse
import paddle
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from models import HashNet
from utils.datasets import get_data, get_dataloader, config_dataset
from utils.datasets import image_transform
from utils.tools import CalcHammingDist, save_database_code

ckp_list = {
    16: "output/weights_16",
    32: "output/weights_32",
    48: "output/weights_48",
    64: "output/weights_64"
}

def get_arguments():
    parser = argparse.ArgumentParser(description='HashNet')
    # normal settings
    parser.add_argument('--model', type=str, default="HashNet")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)

    # Some configs you should change:
    parser.add_argument('--dataset', type=str, default="coco")
    parser.add_argument('--data_path', type=str,
                        default="./datasets/COCO2014/")
                        #NOTE: change to your own data_path!
    parser.add_argument('--img', type=str, default="./resources/COCO_val2014_000000403864.jpg")
    parser.add_argument('--save_path', type=str, default="./output")
    parser.add_argument('--show', action='store_true', default=False, help='show the mateched picture')
    parser.add_argument('--bit', type=int, default=64,
                help="choose the model of certain bit type", choices=[16, 32, 48, 64])
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    arguments = parser.parse_args()
    return arguments

@paddle.no_grad()
def predict(model, image, database_dataset, database_loader, path='./output', bit=64):
    """
    Predict mateched picture for each image.
    """
    # compute bits code for image
    img_code = model(image).sign().numpy()

    # compute bits code for database
    code_path = os.path.join(path, "database_code_{}.npy".format(bit))
    if os.path.isfile(code_path):
        database_code = np.load(code_path)
        print("----- Load code of database from {}".format(code_path))
    else:
        print("----- Not Found: code and label of database!")
        print("----- Ready to compute {} bits code for database".format(bit))
        database_code = save_database_code(model, database_loader, code_path, save=True)
        print("----- Save code and label of database in path: {}".format(code_path))

    # matching
    hamm = CalcHammingDist(database_code, img_code)
    hamm_min = np.min(hamm)
    idx = np.argmin(hamm)
    path, _ = database_dataset.imgs[idx]
    return path, hamm_min

def main(config):
    # define database
    config = config_dataset(config)
    _, _, database_dataset = get_data(config)
    database_loader = get_dataloader(config=config,
                                    dataset=database_dataset,
                                    mode='test')
    # define model
    model = HashNet(config.bit)
    model.eval()

    # load weights
    ckp = ckp_list[config.bit]
    if (ckp).endswith('.pdparams'):
        raise ValueError(f'{ckp} should not contain .pdparams')
    assert os.path.isfile(ckp + '.pdparams'), "{} doesn't exist!".format(ckp + '.pdparams')
    model_state = paddle.load(ckp + '.pdparams')
    model.set_dict(model_state)
    print(f"----- Pretrained: Load model state from {ckp}")

    # define transforms
    eval_transforms = image_transform(config.resize_size, config.crop_size, data_set='test')

    image = Image.open(config.img).convert('RGB')
    img = eval_transforms(image)
    img = img.expand([1] + img.shape)

    pic, hamm_min = predict(model, img, database_dataset, database_loader, config.save_path, config.bit)
    return pic, hamm_min

if __name__ == "__main__":
    config = get_arguments()
    if config.save_path is not None and not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)
    path, hamm_min = main(config)
    print(f"----- Predicted Hamm_min: {hamm_min}")
    print(f"----- Found Mateched Pic: {path}")
    pic = Image.open(path).convert('RGB')
    if config.show:
        pic.show()
    if config.save_path is not None:
        save_path = os.path.join(config.save_path, path.split('/')[-1])
        pic.save(save_path)
        print(f"----- Save Mateched Pic in: {save_path}")
