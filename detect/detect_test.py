import os
import sys
import argparse
import logging
import time
from pathlib import Path
import glob
import json
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils import dataset

import numpy as np
from tqdm import tqdm
import cv2
import yaml

from utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class
from edgetpumodel import EdgeTPUModel

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

    model_old = EdgeTPUModel(args.model, args.labels, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    input_size_old = model_old.get_image_size()

    interpreter = edgetpu.make_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    print(input_zero_point - input_details[0]['quantization'][1])
    print(input_scale - input_details[0]['quantization'][0])
    print(output_zero_point - output_details[0]['quantization'][1])
    print(output_scale - output_details[0]['quantization'][0])

    with open(args.labels, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    labels = data['names']
    logger.info("Loaded {} classes".format(len(labels)))

    size = common.input_size(interpreter)
    
    cam = cv2.VideoCapture(args.device)
    
    while True:
        try:
            res, image = cam.read()
        
            if res is False:
                logger.error("Empty image received")
                break

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(im_rgb, size)
            img = img.astype(np.float32)
            img /= 255.0

            full_image, net_image, pad = get_image_tensor(image, input_size_old[0])
            # print("A- ", img.shape)
            # print("B- ", net_image.shape)
            # print("==========================")
            # print(img-net_image)
            # pred = model_old.forward(net_image)

            common.set_input(interpreter,img)
            interpreter.invoke()
            interpreter_output = interpreter.get_tensor(output_details[0]["index"])
            classes = classify.get_classes(interpreter, top_k=1)

        except KeyboardInterrupt:
            break
        
    cam.release()
            
        

    

