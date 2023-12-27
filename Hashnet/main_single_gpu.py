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

import os, sys, time
import logging
import argparse
import paddle

import warnings
warnings.filterwarnings('ignore')

from utils.loss import HashNetLoss
from utils.datasets import get_data, get_dataloader, config_dataset
from utils.tools import set_random_seed, compute_result, CalcTopMap
from utils.lr_scheduler import DecreaseLRScheduler
from models import HashNet

def get_arguments():
    parser = argparse.ArgumentParser(description='HashNet')
    # normal settings
    parser.add_argument('--model', type=str, default="HashNet")
    parser.add_argument('--bit', type=int, default=16, choices=[16, 32, 48, 64])
    parser.add_argument('--seed', type=int, default=2000, help="NOTE: IMPORTANT TO REPRODUCE THE RESULTS!")
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--pretrained', type=str, default=None, help="checkpoint file")
    parser.add_argument('--resume', type=str, default=None, help="checkpoint file to resume")
    parser.add_argument('--debug_steps', type=int, default=50, help="After each debug_steps, show the train loss")

    # data settings
    parser.add_argument('--dataset', type=str, default="coco_lite")
    parser.add_argument('--data-path', type=str,
                        default="./datasets/COCO2014/", help="data_path")
                        #NOTE: change to your own data_path!
    parser.add_argument('--num_class', type=int, default=80)
    parser.add_argument('--topK', type=int, default=5000)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)

    # training settings
    parser.add_argument('-bs', '--batch-size', type=int, default=64, help="batch_size")
    parser.add_argument('--last_epoch', type=int, default=0)
    parser.add_argument('-ee', '--eval_epoch', type=int, default=10, help="After each eval_epoch, one eval process is performed")
    parser.add_argument('--alpha', type=float, default=0.1, help="Determines the tradeoff between losses")
    parser.add_argument('--step_continuation', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--de_step', type=int, default=50)
    parser.add_argument('-e', '--epoch', type=int, default=150)
    parser.add_argument('-op', '--optimizer', type=str, default="SGD", choices=["SGD", "RMSProp", "AdamW"])
    parser.add_argument('-wd', '--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--save_path', type=str, default="checkpoints/")
    parser.add_argument('--log_path', type=str, default="logs/")
    parser.add_argument('--output-dir', type=str, default="checkpoints/", help='output_dir')
    arguments = parser.parse_args()
    return arguments

def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

def val(model, test_loader, database_loader, config):
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, model)

    # print("calculating database binary code.......")\
    trn_binary, trn_label = compute_result(database_loader, model)

    # print("calculating map.......")
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                        config.topK)
    return mAP

def main():
    config = get_arguments()
    config = config_dataset(config)

    train_dataset, test_dataset, database_dataset = get_data(config)
    config.num_train = len(train_dataset)

    last_epoch = config.last_epoch
    set_random_seed(config.seed)

    if config.eval:
        mode = "eval"
    else:
        mode = "train"

    log_path = '{}/{}-{}-{}'.format(config.log_path, mode, config.bit, time.strftime('%Y%m%d-%H-%M-%S'))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    
    # overall logger
    master_logger = get_logger(
        filename=os.path.join(log_path, 'log.txt'),
        logger_name='master_logger')
    master_logger.info(f'\n{config}')

    model = HashNet(config.bit)

    train_loader = get_dataloader(config=config,
                                  dataset=train_dataset,
                                  mode='train')
    test_loader = get_dataloader(config=config,
                                  dataset=test_dataset,
                                  mode='test')
    database_loader = get_dataloader(config=config,
                                    dataset=database_dataset,
                                    mode='test')

    total_batch_train = len(train_loader)
    total_batch_test = len(test_loader)
    total_batch_base = len(database_loader)
    master_logger.info(f'----- Total # of train batch: {total_batch_train}')
    master_logger.info(f'----- Total # of test batch: {total_batch_test}')
    master_logger.info(f'----- Total # of base batch: {total_batch_base}')

    if config.pretrained:
        if (config.pretrained).endswith('.pdparams'):
            raise ValueError(f'{config.pretrained} should not contain .pdparams')
        assert os.path.isfile(config.pretrained + '.pdparams'), "{} doesn't exist!".format(config.pretrained + '.pdparams')
        model_state = paddle.load(config.pretrained + '.pdparams')
        model.set_dict(model_state)
        master_logger.info(
            "----- Pretrained: Load model state from {}".format(config.pretrained + '.pdparams'))

    if config.eval:
        master_logger.info('----- Start Validating')
        model.eval()
        mAP = val(model, test_loader, database_loader, config)
        master_logger.info("EVAL-{}, bit:{}, dataset:{}, MAP:{:.3f}".format(
                config.model, config.bit, config.dataset, mAP))
        return

    master_logger.info(f"Start training from epoch {last_epoch+1}.")
    config.learning_rate = config.learning_rate * config.batch_size  / 64.0

    scheduler = DecreaseLRScheduler(learning_rate=config.learning_rate,
                                    start_lr=config.learning_rate,
                                    de_step=config.de_step,
                                    last_epoch=last_epoch)
    if config.optimizer == "SGD":
        optimizer = paddle.optimizer.Momentum(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=config.weight_decay,
            momentum=config.momentum)
    elif config.optimizer == "RMSProp":
        optimizer = paddle.optimizer.RMSProp(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=config.weight_decay,
            momentum=config.momentum)
    elif config.optimizer == "AdamW":
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=scheduler,
            weight_decay=config.weight_decay)
    else:
        raise NotImplementedError(f"Unsupported Optimizer: {config.optimizer}.")

    if config.resume:
        assert os.path.isfile(config.resume + '.pdparams') is True
        assert os.path.isfile(config.resume + '.pdopt') is True
        model_state = paddle.load(config.resume + '.pdparams')
        model.set_dict(model_state)
        opt_state = paddle.load(config.resume+'.pdopt')
        optimizer.set_state_dict(opt_state)
        master_logger.info(
                f"----- Resume Training: Load model and optmizer from {config.resume}")

    criterion = HashNetLoss(config.bit, config.num_train, alpha=config.alpha, num_class=config.num_class)

    Best_mAP = 0.
    best_val_epoch = 1
    total_epochs = config.epoch
    for epoch in range(last_epoch + 1, total_epochs + 1):
        model.train()
        criterion.scale = ((epoch - 1) // config.step_continuation + 1) ** 0.5

        train_loss = 0.
        for batch_id, (images, labels, ind) in enumerate(train_loader):
            outputs = model(images)

            loss = criterion(outputs, labels, ind)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if batch_id % config.debug_steps == 0:
                master_logger.info(
                    f"Epoch[{epoch:03d}/{total_epochs:03d}], " +
                    f"Step[{batch_id:04d}/{total_batch_train:04d}], " +
                    f"Loss: {loss.item():.4f}")
        train_loss = train_loss / len(train_loader)

        master_logger.info("{}[{:2d}/{:2d}] bit:{:d}, lr:{:.9f}, scale:{:.3f}, train loss:{:.3f}".format(
            config.model, epoch, total_epochs, config.bit, optimizer.get_lr(), criterion.scale, train_loss))
        scheduler.step()

        if (epoch) % config.eval_epoch == 0 or epoch == total_epochs:
            master_logger.info(f'----- Validation after Epoch: {epoch}')
            model.eval()
            mAP = val(model, test_loader, database_loader, config)

            if mAP > Best_mAP:
                Best_mAP = mAP
                best_val_epoch = epoch
                if config.output_dir is not None:
                    if not os.path.exists(config.output_dir):
                        os.makedirs(config.output_dir)
                    save_path = os.path.join(config.output_dir, "model_best_{}".format(config.bit))
                    master_logger.info(f"save in {save_path}")
                    paddle.save(optimizer.state_dict(), save_path + ".pdopt")
                    paddle.save(model.state_dict(), save_path + ".pdparams")
                    master_logger.info(f'Max mAP so far: {Best_mAP:.4f} at epoch_{best_val_epoch}')
                    master_logger.info(f"----- Save BEST model: {save_path}.pdparams")
                    master_logger.info(f"----- Save BEST optim: {save_path}.pdopt")
            master_logger.info("{} epoch:{}, bit:{}, dataset:{}, MAP:{:.3f}, Best MAP(e{}): {:.3f}".format(
                config.model, epoch, config.bit, config.dataset, mAP, best_val_epoch, Best_mAP))
    master_logger.info("Training completed for {}({}).".format(config.model, config.bit))
    master_logger.info("Best MAP(e{}): {:.3f}".format(best_val_epoch, Best_mAP))


if __name__ == "__main__":
    main()