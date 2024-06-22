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

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_image(image):
    """
    Visualize an image for debug

    Args:
        image: image numpy array, sized (C, H, W)
    """
    # draw image
    draw_img = normalize_image_to_uint8(image)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(image["image"][0, :, :], cmap="gray")
    ax[0].axis("off")
    ax[1].imshow(image["low_res_image"][0, :, :], cmap="gray")
    ax[1].axis("off")
    
    return draw_img

def visualize_image_tf(image):
    """
    Visualize an image for debug

    Args:
        image: image numpy array, sized (C, H, W)
    """
    # draw image
    draw_img = normalize_image_to_uint8(image)
    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)
    
    return draw_img

def print_network_params(params: OrderedDict, show_grad: bool=True):
    """
    Print all network named parama

    Args:
        params: named params from net.named_parameters() dict
        show_grad: show gradient or not
    
    Return:
        None
    """
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