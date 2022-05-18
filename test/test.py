import cv2
import os
import argparse
import logging
import datetime
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Capture")


def run(camera_idx=0, output="None"):

    # Create a VideoCapture object
    logger.info("Starting video stream from camera-"+str(camera_idx))
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
        
    ret, frame = cap.read()

    if ret == True:
        cv2.imwrite("Test.png", frame)

        print("Frame Width: ", frame_width)
        print("Frame Height: ", frame_height)
        print("Saved image size: ", frame.shape)


    # When everything done, release the video capture
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def parse_args():
    # parsing the arguments
    parser = argparse.ArgumentParser("Data Capture")
    parser.add_argument("--output", default="None", type=str, help="USB output directory")
    args = parser.parse_args()
    logger.info(f'Arguements: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args


def main(args):
    run(**vars(args))

if __name__ == '__main__':
    args = parse_args()
    main(args)