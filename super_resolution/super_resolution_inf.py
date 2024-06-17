# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""tienyu
Code for inference pipeline for pre-trained model. Output the evaluation result for obj detection task
Copy and Modify from monai/tutorials/detection.
"""

import argparse
import gc
import json
import logging
import sys
import time
from typing import Any
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from generate_transforms import (
    generate_detection_train_transform,
    generate_detection_val_transform,
    generate_detection_inference_transform,
    generate_detection_train_transform_2d,
    generate_detection_val_transform_2d
)

from torch.utils.tensorboard import SummaryWriter
from visualize_image import visualize_one_xy_slice_in_3d_image,visualize_one_xy_slice_in_2d_image
from warmup_scheduler import GradualWarmupScheduler

import monai
from monai.apps.detection.metrics.coco import COCOMetric
from monai.apps.detection.metrics.matching import matching_batch
#from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.networks.retinanet_network import (
    RetinaNet,
    resnet_fpn_feature_extractor,
)
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape,AnchorGenerator
from monai.data import DataLoader, Dataset, box_utils
from monai.data.utils import no_collation
from monai.transforms import ScaleIntensityRanged
from monai.utils import set_determinism
from monai.networks.nets import ViT


from dataset.load_dataset import load_mednist_datalist
from generate_transforms import generate_mednist_train_transforms, generate_mednist_validation_transforms
from network.autoencoder import Lazy_Autoencoder, Conv_decoder
from visualize_image import visualize_image_tf

def print_network_params(params, show_grad=True):
    v_n,v_v,v_g = [],[],[]
    for name, para in params:
        v_n.append(name)
        v_v.append(para.detach().cpu().numpy() if para is not None else [0])
        if show_grad:
            v_g.append(para.grad.detach().cpu().numpy() if para.grad is not None else [0])
    for i in range(len(v_n)):
        print('value %s: %.3e ~ %.3e (avg: %.3e)'%(v_n[i],np.min(v_v[i]).item(),np.max(v_v[i]).item(),np.mean(v_v[i]).item()))
        if show_grad:
            print('grad %s: %.3e ~ %.3e (avg: %.3e)'%(v_n[i],np.min(v_g[i]).item(),np.max(v_g[i]).item(),np.mean(v_v[i]).item()))

def transform_vitkeys_from_basemodel(state_dict: OrderedDict):
    new_state_dict = OrderedDict()
    params_names = [k for k in state_dict.keys()]
    names_dict = OrderedDict()
    for name in params_names:
        if name.startswith('encoder.'):
            new_name = name
            #not transform encoder_pos_embed
            new_name = new_name.replace('.patch_embed.proj', '.patch_embedding.patch_embeddings')
            new_name = new_name.replace('.fc', '.linear')
            #encoder. => feature_extractor.body.
            new_name = new_name.replace('encoder.', 'feature_extractor.body.')
            new_state_dict[new_name] = state_dict.pop(name)
            names_dict[name] = new_name
    #return
    #print('Transform param name:')
    #print([(k,v) for k, v in names_dict.items()])
    return new_state_dict

class SuperResolutionInference():
    """
    A inference pipeline for pre-trained model in object detection
    Take pre-trained model as input, print & write experimental result

    Args:
        env_dict: envuroment config, include data, model, result path
        config_dict: training config, include training(finetuning) setting for specific dataset / model / gpu
        debug_dict: config for debug, only train or only test
        verbose: bool show details in training or not (for debug),
    Funcs:
        compute: run for inference
        train: training process
        test: testing process
    """
    def __init__(
        self,
        env_dict: dict,
        config_dict: dict,
        debug_dict: dict,
        verbose: bool = False,
    ):
        amp = False
        if amp:
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32
        monai.config.print_config()
        if debug_dict.get('set_deter',False):
            set_determinism(seed=0) #reset determinism for const training!
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(4)

        class_args = argparse.Namespace()
        for k, v in env_dict.items():
            setattr(class_args, k, v)
        for k, v in config_dict.items():
            setattr(class_args, k, v)
        # 1. define transform
        ### !maybe different transform in different dataset other than luna16
        
        train_transforms = generate_mednist_train_transforms()
        val_transforms = generate_mednist_validation_transforms()
        # Use val transform
        inference_transforms = generate_mednist_validation_transforms()
        # 2. prepare training data
        if self.use_train:
            self.make_train_datasets(class_args,train_transforms,val_transforms)
        
        if self.use_test:
            self.make_test_datasets(class_args,inference_transforms)

        #store to self
        self.env_dict, self.config_dict = env_dict, config_dict
        self.args = class_args
        self.verbose = verbose
        self.amp = amp
        self.use_train = debug_dict.get('use_train',True)
        self.use_test = debug_dict.get('use_test',True)
        self.model_name = config_dict.get('model',"retinanet")
    
    def make_train_datasets(self,class_args,train_transforms,val_transforms):
        train_data = load_mednist_datalist(
            class_args.data_list_file_path,
            is_segmentation=True,
            data_list_key="training",
            base_dir=class_args.data_base_dir,
        )
        train_ds = Dataset(
            data=train_data[: int(0.95 * len(train_data))],
            transform=train_transforms,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=7,
            pin_memory=torch.cuda.is_available(),
            collate_fn=no_collation,
            persistent_workers=True,
        )

        # create a validation data loader
        val_ds = Dataset(
            data=train_data[int(0.95 * len(train_data)) :],
            transform=val_transforms,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            collate_fn=no_collation,
            persistent_workers=True,
        )
        self.train_ds, self.val_ds = train_ds, val_ds
        self.train_loader, self.val_loader = train_loader, val_loader
    
    def make_test_datasets(self,class_args,inference_transforms):
        #create a inference data loader
        inference_data = load_mednist_datalist(
            class_args.data_list_file_path,
            is_segmentation=True,
            data_list_key="validation",
            base_dir=class_args.data_base_dir,
        )
        inference_ds = Dataset(
            data=inference_data,
            transform=inference_transforms,
        )
        inference_loader = DataLoader(
            inference_ds,
            batch_size=1,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            collate_fn=no_collation,
        )
        self.inference_ds = inference_ds
        self.inference_loader = inference_loader

    def __call__(self, *args: Any, **kwargs: Any):
        """
        Train and inference based on given pre-trained model. See compute()

        Args:
            *args: positional arguments passed to :func:`compute`
            **kwargs: keyword arguments passed to :func:`compute`

        Returns:
            dict: dictionary with values for evaluation
        """
        return self.compute(*args, **kwargs)
    
    def compute(self, pretrain_network) -> dict:
        """
        Run inference with the `network` pretrained-model.
        1. First build the model and load pre-trained-model
        2. Setup finetune setting
        3. Training (Finetuning)
        4. Evaluation (Testing)
        Args:
            inputs: input of the model inference.
            pre_train_network: model for inference.

        Returns:
            dict: dictionary with values for evaluation (include metric in train and test)
        """
        #1. build the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        metric = torch.nn.MSELoss()
        train_results, test_results, compute_results = {},{},{}
        if self.use_train:
            train_results = self.train(metric=metric, pre_net=pretrain_network, device=device)
        else:
            print("Debug Mode: skip training process with self.use_train = ",self.use_train)
        if self.use_test:
            test_results = self.test(metric=metric, device=device)
        else:
            print("Debug Mode: skip test process with self.use_test = ",self.use_test)
        compute_results['infer_train'] = train_results
        compute_results['infer_test'] = test_results
        return compute_results
    #3. train
    def train(self, anchor_generator, metric, device, pre_net=None):
        """
        Training with the `network` pretrained-model (or not).
        1. First build the model and load pre-trained-model
        2. Setup finetune setting
        3. Training (Finetuning)
        4. Evaluation (Testing)
        Args:
            inputs: input of the model inference.
            pre_train_network: model for inference.

        Returns:
            dict: dictionary with values for evaluation (include metric in train and test)
        """
        # 1-2. build network & load pre-train network
        net = self.build_net(anchor_generator)
        #1-3. load pre-train network !
        if pre_net!=None:
            print('Loaded pretrained model:')
            net.load_state_dict(pre_net, strict=False)
            print_network_params(net.named_parameters(),show_grad=False)

        # 2. Initialize training
        # initlize optimizer, need different version for different setting
        optimizer, scheduler, scaler = self.train_setting_mednist(net)
        loss_func = self.get_loss_func()
        # initialize tensorboard writer
        tensorboard_writer = SummaryWriter(self.args.tfevent_path)
        draw_func = visualize_image_tf
        val_interval = self.config_dict.get('val_interval', 5)  # do validation every val_interval epochs
        best_val_epoch_metric = 1e9
        best_val_epoch = -1  # the epoch that gives best validation metrics
        max_epochs = self.config_dict.get('finetune_epochs', 300)
        epoch_len = len(self.train_ds) // self.train_loader.batch_size
        #torch.autograd.set_detect_anomaly(True) #for debug
        #3. Train
        for epoch in range(max_epochs):
            # ------------- Training -------------
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            net.train()
            epoch_loss = 0
            step = 0
            start_time = time.time()
            scheduler.step()
            # Training
            for batch_data in self.train_loader:
                step += 1
                #flatten targets per image and images per batch
                inputs = batch_data["low_res_image"].to(device)
                targets = batch_data["image"].to(device)

                optimizer.zero_grad(set_to_none=True)
                #with torch.autograd.detect_anomaly(): #for debug
                if self.amp and (scaler is not None):
                    with torch.cuda.amp.autocast():
                        outputs = net(inputs)
                        loss = loss_func(inputs, targets)
                    #with torch.autograd.detect_anomaly(): #for debug
                    scaler.scale(loss).backward()
                    #print_network_params(detector.network.named_parameters())
                    #clip_grad_norm_(detector.network.parameters(), 50) #add grad clip to avoid nan
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = net(inputs)
                    loss = loss_func(inputs, targets)
                    #with torch.autograd.detect_anomaly(): #for debug
                    loss.backward()
                    #print_network_params(detector.network.named_parameters())
                    #clip_grad_norm_(detector.network.parameters(), 50) #add grad clip to avoid nan
                    optimizer.step()
                
                #raise('Stop and debug')
                
                # save to tensorboard
                epoch_loss += loss.detach().item()
                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                tensorboard_writer.add_scalar("train_loss", loss.detach().item(), epoch_len * epoch + step)
                #tmp
                #raise('Stop Training for debug')

            end_time = time.time()
            print(f"Training time: {end_time-start_time}s")
            del batch_data
            torch.cuda.empty_cache()
            gc.collect()

            # save to tensorboard
            epoch_loss /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
            tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)

            # save last trained model
            torch.jit.save(net.network, self.env_dict["model_path"][:-3] + "_last.pt")
            print("saved last model")

            # ------------- Validation for model selection -------------
            if (epoch + 1) % val_interval == 0:
                net.eval()
                val_outputs_all = []
                val_targets_all = []
                epoch_val_loss = 0
                step = 0
                start_time = time.time()
                with torch.no_grad():
                    for val_data in self.val_loader:
                        val_inputs = val_data["low_res_image"].to(device)
                        val_targets = val_data["image"].to(device)
                        if self.amp:
                            with torch.cuda.amp.autocast():
                                val_outputs = net(val_inputs)
                        else:
                            val_outputs = net(val_inputs)

                        # save outputs for evaluation
                        val_outputs_all += val_outputs
                        val_targets_all += val_targets
                        loss = loss_func(val_inputs, val_targets)
                        epoch_val_loss += loss.detach().item()
                        step += 1

                end_time = time.time()
                print(f"Validation time: {end_time-start_time}s")

                # visualize an inference image and boxes to tensorboard
                draw_img = draw_func(
                    ori_image=val_inputs[0][0, ...].cpu().detach().numpy(),
                    image=val_outputs[0][0, ...].cpu().detach().numpy(),
                )
                tensorboard_writer.add_image("val_img", draw_img.transpose([2, 1, 0]), epoch + 1)

                # compute metrics
                del val_inputs
                torch.cuda.empty_cache()
                epoch_val_loss /= step
                print(f"epoch {epoch + 1} validation average loss: {epoch_val_loss:.4f}")
                # write to tensorboard event
                tensorboard_writer.add_scalar("val_loss", epoch_val_loss, epoch + 1)

                # save best trained model
                if epoch_val_loss < best_val_epoch_metric:
                    best_val_epoch_metric = epoch_val_loss
                    best_val_epoch = epoch + 1
                    torch.jit.save(net, self.env_dict["model_path"])
                    print("saved new best metric model")
                print(
                    "current epoch: {} current metric: {:.4f} "
                    "best metric: {:.4f} at epoch {}".format(
                        epoch + 1, epoch_val_loss, best_val_epoch_metric, best_val_epoch
                    )
                )

        print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
        tensorboard_writer.close()
    
    #4. Test
    def test(self, anchor_generator, metric, device,net=None):
        # 2) build test network
        if net==None:
            net = torch.jit.load(self.env_dict["model_path"]).to(device)
            print(f"Load model from {self.env_dict['model_path']}")
        else:
            print(f"Use model from function args")

        loss_func = self.get_loss_func()
        draw_func = visualize_image_tf
        ###! use mscoco evaluation metric noe, mAP,mAR
        # 4. apply trained model
        net.eval()
        epoch_test_loss = 0
        step = 0
        with torch.no_grad():
            start_time = time.time()
            net.eval()
            test_outputs_all = []
            test_targets_all = []
            for test_data in self.inference_loader:
                test_inputs = test_data["low_res_image"].to(device)
                test_targets = test_data["image"].to(device)
                if self.amp:
                    with torch.cuda.amp.autocast():
                        test_outputs = net(test_inputs)
                else:
                    test_outputs = net(test_inputs)

                # save outputs for evaluation
                test_outputs_all += test_outputs
                test_targets_all += test_targets
                loss = loss_func(test_inputs, test_targets)
                epoch_test_loss += loss.detach().item()
                step += 1

        # compute metrics
        del test_inputs
        torch.cuda.empty_cache()
        end_time = time.time()
        print("Testing time: ", end_time - start_time)
            
        epoch_test_loss /= step
        print(f"Test average loss: {epoch_test_loss:.4f}")
        
        test_metric_dict = {'test_mse': epoch_test_loss}
        with open(self.args.result_list_file_path, "w") as outfile:
            json.dump(test_metric_dict, outfile, indent=4)
            
    #build Network
    def build_net(self):
        encoder, decoder = self.build_encoder_decoder()
        print('#'*20)
        print('Build Encoder Network with structure:')
        print_network_params(encoder.named_parameters(),show_grad=False)
        print('#'*20)
        print('Build Decoder Network with structure:')
        print_network_params(decoder.named_parameters(),show_grad=False)
        net = Lazy_Autoencoder(encoder,decoder)
        net = torch.jit.script(net)
        return net
    
    #vitdet
    def build_encoder_decoder(self):
        model_spatial_dims = self.args.spatial_dims
        # Parameter settings are in config.json file
        encoder = ViT(
                in_channels=self.args.n_input_channels, #input channel
                img_size=self.args.img_size,
                patch_size=self.args.model_patch_size,
                hidden_size=self.args.embed_dim,
                mlp_dim=self.args.mlp_dim,
                num_layers=self.args.depth,
                num_heads=self.args.num_heads,
                spatial_dims=model_spatial_dims,
                qkv_bias=True,
                )
        
        decoder = Conv_decoder(
            in_channels=self.args.embed_dim,
            out_channels=1,
            scale_factor=self.args.scale_factor * self.args.model_patch_size, # 4*16 in mednist
            conv_bias=True,
            use_layer_norm=True,
            )
        
        return encoder, decoder

    #training settings
    def train_setting_mednist(self,net):
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        
        after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #150->10
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=after_scheduler)
        scaler = torch.cuda.amp.GradScaler() if self.amp else None
        
        optimizer.zero_grad()
        optimizer.step()
        
        return optimizer, scheduler, scaler
    #loss func
    def get_loss_func(self):
        loss = nn.MSELoss()
        return loss
    
def load_model(path=None,transform_func=None):
    if path:  # make sure to load pretrained model
        if '.ckpt' in path:
            state = torch.load(path, map_location='cpu')
            model = state
        elif '.pth' in path:
            state = torch.load(path, map_location='cpu')
            model = state['state_dict']
        if transform_func!=None:
            model = transform_func(model)
    else:
        model = None
    return model

if __name__ == "__main__":
    '''
    Only for testing all functions in OBJDetectInference.
    set the part what to test (full obj detect, only train, only test, with pre-trained model).
    '''
    #get the config, env, and pre_train network
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print verbose detail during training, recommand True when you are not sure about hyper-parameters",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="pre-trained model path for testing",
    )
    parser.add_argument(
        "-t",
        "--testmode",
        default="full",
        help="which part of func need to test",
    )
    parser.add_argument(
        "-d",
        "--deter",
        default=False,
        action="store_true",
        help="set determinism for model (seed=0)",
    )
    args = parser.parse_args()
    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))
    keys_trans = None
    if config_dict.get("model","")=="vitdet":
        keys_trans = transform_vitkeys_from_basemodel
    pretrained_model = load_model(args.model,keys_trans)
    test_mode = args.testmode
    debug_dict = {} #full test
    if args.testmode=='train': #train func test
        debug_dict['use_test'] = False
    elif args.testmode=='test': #test func test
        debug_dict['use_train'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = OBJDetectInference(env_dict=env_dict, config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)