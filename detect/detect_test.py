import os
import argparse
import logging
import numpy as np
import cv2
import yaml

from pycoral.utils import edgetpu
from pycoral.adapters import common

from objects import get_objects

def make_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3):

    lw = line_width or max(int(min(im.size) / 200), 2)

    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    
    cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)
        txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        c2 = c1[0] + txt_width, c1[1] - txt_height - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    
    return im

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


def main():
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
    top = 5

        
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
                    n = (nms_result[0][:, -1] == c).sum()
                    s += f"{n} {labels[int(c)]}{'s' * (n > 1)}, "
                
                if s != "":
                    s = s.strip()
                    s = s[:-1]
                
                logger.info("Detected: {}".format(s))

                for *xyxy, conf, cls in reversed(nms_result[0]):
                    c = int(cls)
                    label = f'{labels[c]} {conf:.2f}'
                    output_image = make_box(xyxy, image, label=label)

                cv2.imshow('frame', output_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            break
        
    cam.release()

if __name__ == '__main__':
    main()