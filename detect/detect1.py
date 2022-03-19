import cv2
import os

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


try:
    # Try importing the small tflite_runtime module (this runs on the Dev Board)
    print("Trying to import tensorflow lite runtime...")
    from tflite_runtime.interpreter import Interpreter, load_delegate
    experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
except ModuleNotFoundError:
    # Try importing the full tensorflow module (this runs on PC)
    try:
        print("TFLite runtime not found; trying to import full tensorflow...")
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        experimental_delegates = None
    except ModuleNotFoundError:
        # Couldn't import either module
        raise RuntimeError("Could not import Tensorflow or Tensorflow Lite")



def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def main():
    default_model_dir = '../yolo_model'
    default_model = 'yolov5s-int8_edgetpu.tflite'
    default_labels = 'coco_labels.txt'

    model = os.path.join(default_model_dir, default_model)
    labels = os.path.join(default_model_dir, default_labels)

    # tfl_filename = "lstm_mnist_model_b10000.tflite"
    interpreter = Interpreter(model_path=model,
    experimental_delegates=experimental_delegates)
    interpreter.allocate_tensors()


    # interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    vc = cv2.VideoCapture(0)

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        print("Hi")
        print(interpreter.get_output_details())
        objs = get_objects(interpreter, 0.1)[:3]
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
