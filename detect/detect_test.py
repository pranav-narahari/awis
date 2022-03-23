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
from pycoral.adapters.detect import get_objects

import numpy as np
from tqdm import tqdm
import cv2
import yaml

import utils
from objects import get_objects

def get_BBox(xyxy, output_image, size):

    in_h, in_w = size
    out_h, out_w, _ = output_image.shape
            
    ratio_w = out_w/(in_w)
    ratio_h = out_h/(in_h) 
    
    out = []
    for coord in xyxy:

        x1, y1, x2, y2 = coord
                    
        x1 *= in_w*ratio_w
        x2 *= in_w*ratio_w
        y1 *= in_h*ratio_h
        y2 *= in_h*ratio_h
        
        x1 = max(0, x1)
        x2 = min(out_w, x2)
        
        y1 = max(0, y1)
        y2 = min(out_h, y2)
        
        out.append((x1, y1, x2, y2))
    
    return np.array(out).astype(int)

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

    conf_thresh = 0.5
    iou_thresh = 0.45
    top = 3

        
    logger.info("Opening stream on device: {}".format(args.device))

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
                break

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(im_rgb, size)
            img = img.astype(np.float32)
            img /= 255.0


            if img.shape[0] == 3:
                img = img.transpose((1,2,0))
        
            img = img.astype('float32')

            # Scale input, conversion is: real = (int_8 - zero)*scale
            img = (img/input_scale) + input_zero_point
            img = img[np.newaxis].astype(np.uint8)

            common.set_input(interpreter,img)
            interpreter.invoke()
            interpreter_output = interpreter.get_tensor(output_details[0]["index"])
            result = output_scale * (interpreter_output.astype('float32') - output_zero_point)
            nms_result = get_objects(result, conf_thresh, iou_thresh, top)


            if len(nms_result[0]):
                nms_result[0][:, :4] = get_BBox(nms_result[0][:,:4], image, size)

                
                s = ""
                
                # Print results
                for c in np.unique(nms_result[0][:, -1]):
                    n = (nms_result[0][:, -1] == c).sum()  # detections per class
                    s += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                if s != "":
                    s = s.strip()
                    s = s[:-1]
                
                logger.info("Detected: {}".format(s))

                for *xyxy, conf, cls in reversed(nms_result[0]):
                    c = int(cls)  # integer class
                    label = f'{labels[c]} {conf:.2f}'
                    output_image = utils.plot_one_box(xyxy, image, label=label)

                cv2.imshow('frame', output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            break
        
    cam.release()
            
        

    

