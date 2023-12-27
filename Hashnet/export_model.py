# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
import argparse
import paddle
from paddle.static import InputSpec

from models import HashNet

ckp_list = {
    16: "output/weights_16.pdparams",
    32: "output/weights_32.pdparams",
    48: "output/weights_48.pdparams",
    64: "output/weights_64.pdparams"
}

def get_arguments():
    parser = argparse.ArgumentParser(description='Model export')
    parser.add_argument('--model', type=str, default="HashNet")
    parser.add_argument('--bit', type=int, default=64, choices=[16, 32, 48, 64])
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--save-inference-dir', type=str, default="./tipc_output/", help='save_inference_dir')
    parser.add_argument('--load_ckp', action='store_false', default=True)
    parser.add_argument('--pretrained', type=str, default=None,
                help='If pretrained is None, model load from ckp_list')
    arguments = parser.parse_args()
    return arguments


def main(args):
    model = HashNet(args.bit)

    if args.load_ckp:
        if args.pretrained is not None:
            if (args.pretrained).endswith('.pdparams'):
                raise ValueError(f'{args.pretrained} should not contain .pdparams')
            assert os.path.isfile(args.pretrained + '.pdparams'), "{} doesn't exist!".format(args.pretrained + '.pdparams')
            para_state_dict = paddle.load(args.pretrained + '.pdparams')
        else:
            ckp = ckp_list[args.bit]
            para_state_dict = paddle.load(ckp)
        model.set_dict(para_state_dict)
        print('Loaded trained params of model({}) successfully.'.format(args.bit))

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])

    # save inference model
    save_path = os.path.join(args.save_inference_dir, "inference_{}".format(args.bit))
    paddle.jit.save(model, save_path)
    print(f"inference model is saved in {save_path}")


if __name__ == "__main__":
    args = get_arguments()
    main(args)