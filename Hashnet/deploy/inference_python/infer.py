# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image
from reprod_log import ReprodLogger
from paddle.vision.transforms import transforms

import sys
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from utils.datasets import get_data, config_dataset
from utils.tools import CalcHammingDist

class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference_{}.pdmodel".format(self.args.bit)),
            os.path.join(args.model_dir, "inference_{}.pdiparams".format(self.args.bit)))

        # build transforms
        self.transforms = transforms.Compose([
                transforms.Resize(self.args.resize_size),
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        print(self.args)
        # define database
        self.args = config_dataset(self.args)
        print(self.args)
        _, _, self.database_dataset = get_data(self.args)

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.crop_size,
                                   self.args.crop_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()
        return

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, output):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns:
            hamm_min: minimum CalcHammingDist,
            path: the corresponding idx of picture in database.
            hamm: CalcHammingDist
        """
        # compute bits code for image
        img_code = np.sign(output[0])
        img_code = np.expand_dims(img_code, axis=0)

        # compute bits code for database
        code_path = os.path.join(self.args.save_path, "database_code_{}.npy".format(self.args.bit))
        if os.path.isfile(code_path):
            database_code = np.load(code_path)
        else:
            if not os.path.exists(self.args.save_path):
                os.makedirs(self.args.save_path, exist_ok=True)
            database_code = self.save_database_code(code_path)
        print("----- Load code of database from {}".format(code_path))

        # matching
        hamm = CalcHammingDist(database_code, img_code)
        hamm_min = np.min(hamm)
        idx = np.argmin(hamm)
        path, _ = self.database_dataset.imgs[idx]
        return path, hamm_min, hamm

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output

    def save_database_code(self, code_path, save=True):
        bs = []
        from tqdm import tqdm
        for idx in tqdm(range(len(self.database_dataset)), desc="Computing Database Code"):
            path, _ = self.database_dataset.imgs[idx]
            img = self.preprocess(path)
            bs.append(self.run(img))
        database_code =  np.sign(np.concatenate(bs))
        if save:
            np.save(code_path, database_code)
            print("----- Save code and label of database in path: {}".format(code_path))
        return database_code


def get_args(add_help=True):
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle", add_help=add_help)
    parser.add_argument('--bit', type=int, default=64,
                help="choose the model of certain bit type", choices=[16, 32, 48, 64])
    parser.add_argument('--dataset', type=str, default="coco")
    parser.add_argument('--data-path', type=str,
                        default="./datasets/COCO2014/", help="data_path")
                        #NOTE: change to your own data_path!
    parser.add_argument('--save-path', type=str,
                        default="./output", help="save_path, where to save database_code_bits.npy")
    parser.add_argument(
        "--model-dir", default='./tipc_output', help="inference model dir, model_dir")
    parser.add_argument(
        "--use-gpu", default=True, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--max-batch-size", default=16, type=int, help="max_batch_size")
    parser.add_argument("--batch-size", default=1, type=int, help="batch_size")
    parser.add_argument("--crop-size", default=224, type=int, help="crop_szie")
    parser.add_argument("--resize-size", default=256, type=int, help="resize_szie")
    parser.add_argument("--img-path", default="./resources/COCO_val2014_000000403864.jpg", help="img_path")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")

    args = parser.parse_args()
    return args


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    # init inference engine
    inference_engine = InferenceEngine(args)

    # init benchmark log
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="example",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img = inference_engine.preprocess(args.img_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    path, hamm_min, hamm = inference_engine.postprocess(output)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()
    return path, hamm_min, hamm


if __name__ == "__main__":
    args = get_args()
    path, hamm_min, hamm = infer_main(args)
    print(f"----- Predicted Hamm_min: {hamm_min}")
    print(f"----- Found Mateched Pic: {path}")
    reprod_logger = ReprodLogger()
    reprod_logger.add("hamm_min", np.array([hamm_min]))
    reprod_logger.add("hamm", np.array([hamm]))
    reprod_logger.save("output_inference_engine.npy")
