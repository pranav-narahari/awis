import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json
from pycoral.utils import edgetpu

import numpy as np
from tqdm import tqdm
import cv2
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    default_model_dir = '../yolo_model'
    default_model = 'yolov5s-int8-224_edgetpu.tflite'
    default_labels = 'coco.yaml'
    parser = argparse.ArgumentParser("EdgeTPU test runner")
    parser.add_argument("--model", "-m", help="weights file", 
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="model confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--labels", type=str, help="Labels file", 
                        default=os.path.join(default_model_dir,default_labels))
    parser.add_argument("--device", type=int, default=0, help="Image capture device to run live detection")
   
    args = parser.parse_args()

    conf_thresh = 0.25
    iou_thresh = 0.45

        
    logger.info("Opening stream on device: {}".format(args.device))
    
    cam = cv2.VideoCapture(args.device)

    interpreter = edgetpu.make_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    with open(args.labels, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    print(data)
    labels = data['names']
    logger.info("Loaded {} classes".format(len(labels)))
    
    while True:
        try:
            res, image = cam.read()
        
            if res is False:
                logger.error("Empty image received")
                break

            interpreter = edgetpu.make_interpreter(args.model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

        except KeyboardInterrupt:
            break
        
    cam.release()
            
        

    

