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
import yaml
import time
from typing import Any
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.utils.tensorboard import SummaryWriter
from network.warmup_scheduler import GradualWarmupScheduler
import monai

from monai.data import DataLoader, Dataset
from monai.utils import set_determinism
from monai.networks.nets import ViT
from monai.losses import PerceptualLoss

from dataset.load_dataset import load_mednist_datalist,load_eyeq_datalist
from utils.transform.superresolution import generate_train_transforms, generate_validation_transforms
from network.autoencoder import Lazy_Autoencoder, Conv_decoder
from utils.visualize import visualize_image_tf, print_network_params
from utils.utils import load_model
from utils.evaluation.superresolution_metric import PSNR, SSIM

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
        amp = config_dict.get('amp',False)
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
        
        #store to self
        self.env_dict, self.config_dict = env_dict, config_dict
        self.args = class_args
        self.verbose = verbose
        self.amp = amp
        self.use_train = debug_dict.get('use_train',True)
        self.use_test = debug_dict.get('use_test',True)
        self.model_name = config_dict.get('model',"retinanet")
        # 1. define transform
        ### !maybe different transform in different dataset other than luna16
        low_res_size = int(self.args.img_size // self.args.scale_factor)
        turn2gray = (self.config_dict.get("data_channels",1)!=1)
        train_transforms = generate_train_transforms(image_size=self.args.img_size,lowres_img_size=low_res_size,to_gray=turn2gray)
        val_transforms = generate_validation_transforms(image_size=self.args.img_size,lowres_img_size=low_res_size,to_gray=turn2gray)
        # Use val transform
        inference_transforms = generate_validation_transforms(image_size=self.args.img_size,lowres_img_size=low_res_size,to_gray=turn2gray)
        # 2. prepare training data
        if self.use_train:
            self.make_train_datasets(class_args,train_transforms,val_transforms)
        
        if self.use_test:
            self.make_test_datasets(class_args,inference_transforms)
    
    def make_train_datasets(self,class_args,train_transforms,val_transforms):
        if self.args.dataset=="mednist":
            load_data_func = load_mednist_datalist
        elif self.args.dataset=="eyeq":
            load_data_func = load_eyeq_datalist
        
        train_data = load_data_func(
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
            persistent_workers=True,
        )

        # create a validation data loader
        val_ds = Dataset(
            data=train_data[int(0.95 * len(train_data)) :],
            transform=val_transforms,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
        )
        self.train_ds, self.val_ds = train_ds, val_ds
        self.train_loader, self.val_loader = train_loader, val_loader
    
    def make_test_datasets(self,class_args,inference_transforms):
        #create a inference data loader
        if self.args.dataset=="mednist":
            load_data_func = load_mednist_datalist
        elif self.args.dataset=="eyeq":
            load_data_func = load_eyeq_datalist
        
        inference_data = load_data_func(
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
        ## !! need handle different scale problem
        metric_dic = {
            'PSNR': PSNR(data_range=1.0, device=device),
            'SSIM': SSIM(device=device),
            }
        train_results, test_results, compute_results = {},{},{}
        if self.use_train:
            train_results = self.train(metric_dic=metric_dic, pre_net=pretrain_network, device=device)
        else:
            print("Debug Mode: skip training process with self.use_train = ",self.use_train)
        if self.use_test:
            test_results = self.test(metric_dic=metric_dic, device=device)
        else:
            print("Debug Mode: skip test process with self.use_test = ",self.use_test)
        compute_results['infer_train'] = train_results
        compute_results['infer_test'] = test_results
        return compute_results
    #3. train
    def train(self, metric_dic, device, pre_net=None):
        """
        Training with the `network` pretrained-model (or not).
        1. First build the model and load pre-trained-model
        2. Setup finetune setting
        3. Training (Finetuning)
        4. Evaluation (Testing)
        Args:
            metric_dic: dict for {metric_name: metric_func}
            device
            pre_net: pre-train model
        Save model to self.env_dict["model_path"]
        """
        # 1-2. build network & load pre-train network
        net = self.build_net().to(device)
        #1-3. load pre-train network !
        if pre_net!=None:
            print('Loaded pretrained model:')
            net.load_state_dict(pre_net, strict=False)
            print_network_params(net.named_parameters(),show_grad=False)

        # 2. Initialize training
        # initlize optimizer, need different version for different setting
        optimizer, scheduler, scaler = self.train_setting(net)
        loss_func_dic = self.get_loss_func_dic(device=device)
        # initialize tensorboard writer
        tensorboard_writer = SummaryWriter(self.args.tfevent_path)
        draw_func = visualize_image_tf
        val_interval = self.config_dict.get('val_interval', 5)  # do validation every val_interval epochs
        best_val_epoch_metric = -1e9
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
            each_loss_dic = {k:0 for k in loss_func_dic.keys()}
            step = 0
            start_time = time.time()
            scheduler.step()
            # Training
            for batch_data in self.train_loader:
                step += 1
                #flatten targets per image and images per batch
                #print(batch_data)
                inputs = batch_data["low_res_image"].to(device)
                targets = batch_data["image"].to(device)
                #print('low res img shape: ', inputs.shape)
                #print('ori img shape: ',targets.shape)

                optimizer.zero_grad(set_to_none=True)
                #with torch.autograd.detect_anomaly(): #for debug
                loss = 0
                losses_num = 0
                
                if self.amp and (scaler is not None):
                    with torch.cuda.amp.autocast():
                        outputs = net(inputs)
                        for k,loss_func in loss_func_dic.items():
                            each_loss = loss_func(outputs, targets)
                            loss += each_loss
                            each_loss_dic[k] += each_loss.detach().cpu().item()
                            losses_num+=1
                        loss = loss / losses_num
                    #with torch.autograd.detect_anomaly(): #for debug
                    scaler.scale(loss).backward()
                    #print_network_params(detector.network.named_parameters())
                    #clip_grad_norm_(detector.network.parameters(), 50) #add grad clip to avoid nan
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = net(inputs)
                    for k,loss_func in loss_func_dic.items():
                        each_loss = loss_func(outputs, targets)
                        loss += each_loss
                        each_loss_dic[k] += each_loss.detach().cpu().item()
                        losses_num+=1
                    loss = loss / losses_num
                    #with torch.autograd.detect_anomaly(): #for debug
                    loss.backward()
                    #print_network_params(detector.network.named_parameters())
                    #clip_grad_norm_(detector.network.parameters(), 50) #add grad clip to avoid nan
                    optimizer.step()
                #print('outputs shape: ', outputs.shape)
                #raise('Stop and debug')
                
                # save to tensorboard
                epoch_loss += loss.detach().item()
                print(f"{step}/{epoch_len}, train_loss: {loss.detach().item():.4f}")
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
            for k in each_loss_dic.keys():
                each_loss_dic[k] /= step
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            tensorboard_writer.add_scalar("avg_train_loss", epoch_loss, epoch + 1)
            tensorboard_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch + 1)
            print(', '.join([f'{k} loss: {v:.4f}' for k,v in each_loss_dic.items()]))
            for k,v in each_loss_dic.items():
                tensorboard_writer.add_scalar(f"avg_{k}_loss", v, epoch + 1)

            # save last trained model
            torch.jit.save(net, self.env_dict["model_path"][:-3] + "_last.pt")
            print("saved last model")

            # ------------- Validation for model selection -------------
            if (epoch + 1) % val_interval == 0:
                net.eval()
                epoch_metric_val = {k:0 for k in metric_dic.keys()}
                each_loss_dic = {k:0 for k in loss_func_dic.keys()}
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
                        for k,metric in metric_dic.items():
                            metric_val = metric(val_outputs, val_targets)
                            epoch_metric_val[k] += metric_val.detach().item()
                        for k,loss_func in loss_func_dic.items():
                            each_loss = loss_func(val_outputs, val_targets)
                            each_loss_dic[k] += each_loss.detach().cpu().item()
                        step += 1

                end_time = time.time()
                print(f"Validation time: {end_time-start_time}s")
                del val_data
                torch.cuda.empty_cache()
                gc.collect()
                # visualize an inference image and boxes to tensorboard
                draw_img_ori = draw_func(val_targets[0].permute(1,2,0).cpu().detach().numpy())
                draw_img = draw_func(val_outputs[0].permute(1,2,0).cpu().detach().numpy())
                tensorboard_writer.add_image("val_output_img", draw_img.transpose([2, 1, 0]), epoch + 1)
                tensorboard_writer.add_image("val_origin_img", draw_img_ori.transpose([2, 1, 0]), epoch + 1)

                # compute metrics & write to tensorboard event
                del val_inputs
                torch.cuda.empty_cache()
                for k in epoch_metric_val.keys():
                    epoch_metric_val[k] /= step
                    print(f"epoch {epoch + 1} validation {k}: {epoch_metric_val[k]:.4f}")
                    tensorboard_writer.add_scalar(f"val_{k}", epoch_metric_val[k], epoch + 1)
                for k in each_loss_dic.keys():
                    each_loss_dic[k] /= step
                    tensorboard_writer.add_scalar(f"val_{k}_loss", v, epoch + 1)
                print(', '.join([f'{k} loss: {v:.4f}' for k,v in each_loss_dic.items()]))

                # save best trained model
                epoch_val_sum = 0
                for k,v in epoch_metric_val.items():
                    if 'loss' in k:
                        epoch_val_sum = epoch_val_sum + (1-v)
                    else:
                        epoch_val_sum += v
                if epoch_val_sum > best_val_epoch_metric:
                    best_val_epoch_metric = epoch_val_sum
                    best_val_epoch = epoch + 1
                    torch.jit.save(net, self.env_dict["model_path"])
                    print("saved new best metric model")
                print(
                    "current epoch: {} current metric: {:.4f} "
                    "best metric: {:.4f} at epoch {}".format(
                        epoch + 1, epoch_val_sum, best_val_epoch_metric, best_val_epoch
                    )
                )

        print(f"train completed, best_metric: {best_val_epoch_metric:.4f} " f"at epoch: {best_val_epoch}")
        tensorboard_writer.close()
    
    #4. Test
    def test(self, metric_dic, device,net=None):
        """
        Testing
        Args:
            metric_dic: dict for {metric_name: metric_func}
            device
            net: trained model for inference.

        Returns:
            dict: dictionary with values for evaluation (include metric in train and test)
        """
        # 2) build test network
        if net==None:
            net = torch.jit.load(self.env_dict["model_path"]).to(device)
            print(f"Load model from {self.env_dict['model_path']}")
        else:
            print(f"Use model from function args")

        ###! use mscoco evaluation metric noe, mAP,mAR
        # 4. apply trained model
        net.eval()
        epoch_metric_val = {k:0 for k in metric_dic.keys()}
        step = 0
        with torch.no_grad():
            start_time = time.time()
            net.eval()
            for test_data in self.inference_loader:
                test_inputs = test_data["low_res_image"].to(device)
                test_targets = test_data["image"].to(device)
                if self.amp:
                    with torch.cuda.amp.autocast():
                        test_outputs = net(test_inputs)
                else:
                    test_outputs = net(test_inputs)

                # save outputs for evaluation
                for k,metric in metric_dic.items():
                    metric_val = metric(test_outputs, test_targets)
                    epoch_metric_val[k] += metric_val.detach().item()
                step += 1

        # compute metrics
        del test_inputs
        torch.cuda.empty_cache()
        end_time = time.time()
        print("Testing time: ", end_time - start_time)
            
        for k in epoch_metric_val.keys():
            epoch_metric_val[k] /= step
            print(f"Test average {k}: {epoch_metric_val[k]:.4f}")
        
        test_metric_dict = {f'test_{k}': v for k,v in epoch_metric_val.items()}
        with open(self.args.result_list_file_path, "w") as outfile:
            json.dump(test_metric_dict, outfile, indent=4)
        return test_metric_dict
            
    #build Network
    def build_net(self):
        '''
        Build Autoencoder Network
        Returns:
            net
        '''
        encoder, decoder = self.build_encoder_decoder()
        total_scale_factor = int(self.args.scale_factor * self.args.model_patch_size)
        latent_size = int(self.args.img_size / total_scale_factor)
        latent_shape = [-1, self.args.embed_dim] + [latent_size for i in range(self.args.spatial_dims)]
        net = Lazy_Autoencoder(encoder,decoder, latent_img_shape=latent_shape)
        net = torch.jit.script(net)
        print('#'*20)
        print('Build Network with structure:')
        print_network_params(net.named_parameters(),show_grad=False)
        return net
    
    #vitdet
    def build_encoder_decoder(self):
        '''
        Build Encoder and Decoder
        Returns:
            encoder
            decoder
        '''
        model_spatial_dims = self.args.spatial_dims
        total_scale_factor = int(self.args.scale_factor * self.args.model_patch_size)
        low_resol_img_size = int(self.args.img_size // self.args.scale_factor)
        # Parameter settings are in config.json file
        encoder = ViT(
                in_channels=self.args.n_input_channels, #input channel
                img_size=low_resol_img_size,
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
            out_channels=self.args.n_input_channels,
            scale_factor=total_scale_factor, # 4*16 in mednist
            conv_bias=True,
            use_layer_norm=True,
            )
        
        return encoder, decoder

    #training settings
    def train_setting(self,net):
        '''
        Create optimizer, scheduler, and scaler based on config
        Returns:
            optimizer
            scheduler
            scaler
        '''
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        
        after_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.warmup_epochs, after_scheduler=after_scheduler)
        scaler = torch.cuda.amp.GradScaler() if self.amp else None
        
        optimizer.zero_grad()
        optimizer.step()
        
        return optimizer, scheduler, scaler
    #loss func
    def get_loss_func_dic(self,device):
        '''
        Returns:
            dict: ['percep': PerceptualLoss, 'l1': nn.L1Loss, 'mse': nn.MSELoss]
        '''
        loss_1 = PerceptualLoss(spatial_dims=self.args.spatial_dims).to(device)
        loss_2 = nn.L1Loss().to(device)
        loss_3 = nn.MSELoss().to(device)
        return {'percep': loss_1, 'l1': loss_2, 'mse': loss_3}
    
if __name__ == "__main__":
    #get the config, env, and pre_train network
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.yaml",
        help="environment yaml file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train.yaml",
        help="config yaml file that stores hyper-parameters",
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
    env_dict = yaml.safe_load(open(args.environment_file, "r"))
    config_dict = yaml.safe_load(open(args.config_file, "r"))
    transform_dic = {
        '.patch_embed.proj': '.patch_embedding.patch_embeddings', 
        '.fc': '.linear',
    }
    pretrained_model = load_model(args.model,transform_dic)
    test_mode = args.testmode
    debug_dict = {} #full test
    if args.testmode=='train': #train func test
        debug_dict['use_test'] = False
    elif args.testmode=='test': #test func test
        debug_dict['use_train'] = False
    if args.deter:
        debug_dict["set_deter"] = True
    #
    inferer = SuperResolutionInference(env_dict=env_dict, config_dict=config_dict, debug_dict=debug_dict, verbose=args.verbose)
    inferer.compute(pretrain_network=pretrained_model)