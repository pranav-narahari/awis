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

import utils
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

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    if input_scale < 1e-9:
        input_scale = 1.0

    if output_scale < 1e-9:
        output_scale = 1.0


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



            common.set_input(interpreter,img)
            interpreter.invoke()
            interpreter_output = interpreter.get_tensor(output_details[0]["index"])
            result = output_scale * (interpreter_output.astype('float32') - output_zero_point)

            full_image, net_image, pad = utils.get_image_tensor(image, input_size_old[0])
            # pred = model_old.forward(net_image)

            # result = pred

            if len(result[0]):
                # Rescale boxes from img_size to im0 size
                # x1, y1, x2, y2=
                result[0][:, :4] = utils.get_scaled_coords(result[0][:,:4], image, pad, size)

                
                s = ""
                
                # Print results
                for c in np.unique(result[0][:, -1]):
                    n = (result[0][:, -1] == c).sum()  # detections per class
                    s += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                if s != "":
                    s = s.strip()
                    s = s[:-1]
                
                logger.info("Detected: {}".format(s))

        except KeyboardInterrupt:
            break
        
    cam.release()
            
        

    

