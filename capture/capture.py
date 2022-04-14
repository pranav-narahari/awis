import cv2
import os
import argparse
import logging
import datetime
import time
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Capture")

class camThread(threading.Thread):
    def __init__(self, delay, camera_idx, output):
        threading.Thread.__init__(self)
        self.delay = delay
        self.camera_idx = camera_idx
        self.output = output
    def run(self):
        action(self.delay, self.camera_idx, self.output)



def action(delay=1, camera_idx=0, output="None"):

    #assign output directorie
    image_path = output

    # Create a VideoCapture object
    logger.info("Starting video stream from camera-"+str(camera_idx))
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,300)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    image = True
    capture = True

    start_time = time.time() #start time for delay
    while(True):
        
        ret, frame = cap.read()

        if ret == True:

            #Image recording
            if image and capture:
                if time.time()-start_time >= delay:
                    image_name = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+"_camera_"+str(camera_idx)+".jpeg")
                    cv2.imwrite(os.path.join(image_path,image_name), frame)
                    logger.info(f'Image saved to {os.path.join(image_path,image_name)}')
                    start_time = time.time()

            # Display the resulting frame
            # cv2.imshow('frame',frame)
            
            # Press r on keyboard to start/stop image capture
            if cv2.waitKey(1) & 0xFF == ord('c'):
                capture = not capture
                logger.info(f'Image Capture with {delay}s delay in Progress' if capture else f'Image Capture Ended')
            
            
            # Press q on keyboard to quit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info(f'Exiting')
                break

        # Break the loop
        else:
            break  

    # When everything done, release the video capture
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def parse_args():
    # parsing the arguments
    parser = argparse.ArgumentParser("Data Capture")
    parser.add_argument("--delay", type=int, default=1, help="image capture delay")
    parser.add_argument('--camera_idx1', type=int, default=0, help='Index of which video source to use')
    parser.add_argument('--camera_idx2', type=int, default=-1, help='Index of which video source to use')
    parser.add_argument("--output", default="None", type=str, help="USB output directory")
    args = parser.parse_args()
    logger.info(f'Arguements: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args

def main(args):
    if args.camera_idx2 != -1:
        thread1 = camThread(args.delay, args.camera_idx1, args.output)
        thread2 = camThread(args.delay, args.camera_idx2, args.output)
        thread1.start()
        thread2.start()
    else:
        action(args.delay, args.camera_idx1, args.output)

if __name__ == '__main__':
    args = parse_args()
    main(args)