import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json

import numpy as np
from tqdm import tqdm
import cv2
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from edgetpumodel import EdgeTPUModel
from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class

if __name__ == "__main__":
 
    default_model_dir = '../yolo_model'
    default_model = 'yolov5s-int8-224_edgetpu.tflite'
    default_labels = 'coco.yaml'
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", 
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--names", type=str, help="Labels file", 
                        default=os.path.join(default_model_dir,default_labels))
    parser.add_argument("--device", type=int, default=0, help="Image capture device to run live detection")


    args = parser.parse_args()

    model = EdgeTPUModel(args.model, args.names, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    input_size = model.get_image_size()

    conf_thresh = 0.25
    iou_thresh = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
        

    logger.info("Opening stream on device: {}".format(args.device))
    
    cam = cv2.VideoCapture(args.device)
    
    while True:
        try:
            res, image = cam.read()
        
            if res is False:
                logger.error("Empty image received")
                break
            else:
                full_image, net_image, pad = get_image_tensor(image, input_size[0])
                pred = model.forward(net_image)
            
                out_im = model.process_predictions(pred[0], full_image, pad)
            
                tinference, tnms = model.get_last_inference_time()
                logger.info("Frame done in {}".format(tinference+tnms))
                
                cv2.imshow('frame', out_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            break
        
    cam.release()
