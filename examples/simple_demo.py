# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:29:26 2019

@author: nsde
"""

#%%
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from libcpab import Cpab
from libcpab.core.utility import show_images, get_dir # utility functions

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import time 


#%%
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='numpy', 
                        choices=['numpy', 'tensorflow', 'pytorch'],
                        help='backend to run demo with')
    parser.add_argument('--device', default='cpu',
                        choices=['cpu', 'gpu'],
                        help='device to run demo on')
    return parser.parse_args()

#%%
if __name__ == "__main__":
    args = argparser()
    print("---Running script with arguments---")
    print("\n".join([str(k) + ':' + str(v) for k,v in vars(args).items()]))
    print("-----------------------------------")
    
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    
    # Number of transformed samples 
    N = 9
    time1  = time.time()
    
    # Load some data
    data = plt.imread(get_dir(__file__) + '/../data/cat.jpg') 
    data = np.tile(data[None], [N,1,1,1]) # create batch of data
    time2 = time.time()
    print("--- loading file time ",time2-time1)
    # Create transformer class
    time3 = time.time()
    T = Cpab([3, 3], backend=args.backend, device=args.device, 
             zero_boundary=True, volume_perservation=False, override=False)
    time4 = time.time()
    print("--- loading Cpab class ",time4-time3)

    # Sample random transformation
    time5  = time.time()
    theta  = T.sample_transformation(N)
    time6  = time.time()
    print("--- getting sample transformation ",time6-time5)

    # Convert data to the backend format
    time7 = time.time()
    data  = T.backend.to(data, device=args.device)
    time8 = time.time()
    print("---- transfer data from cpu to gpu ",time8-time7)

    # Pytorch have other data format than tensorflow and numpy, color 
    # information is the second dim. We need to correct this before and after
    time9 = time.time()
    data = data.permute(0,3,1,2) if args.backend=='pytorch' else data
    time10 = time.time()

    print("---- change shape of data ",time10-time9)
    # Transform the images
    time11 = time.time()

    t_data = T.transform_data(data, theta, outsize=(350, 350))

    time12  = time.time()
    
    print("--- final transformation",time12-time11)

    time13 = time.time()


    # Get the corresponding numpy arrays in correct format
    t_data = t_data.permute(0,2,3,1) if args.backend=='pytorch' else t_data
    time14 = time.time()

    print("--- changing back shape from pytorch",time14-time13)

    time15  = time.time()
    t_data = T.backend.tonumpy(t_data)
    time16 = time.time()
    print("--- back to numpy format",time16-time15)
    print(len(t_data))
    counter_ = 1
    for data_ in t_data:
        succ = cv2.imwrite("output2_{}.png".format(counter_),data_)
        print(succ)
        counter_+=1
    # Show transformed samples
#     show_images(t_data)
