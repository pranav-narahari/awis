import cv2
import os
import argparse
import logging
import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Capture")

def create_dir(usb):

    #setting folder paths
    video_folder = "VideoData"
    image_folder = "ImageData"

    #save in USB or local
    if usb == 'None':
        output_dir = "../data"
    else:
        #Checking USB disk 
        if not (os.path.exists(os.path.join("/Volumes",usb)) or os.path.exists(os.path.join("/media",os.getlogin(),usb))):
            logger.error("USB Not Connected")
            exit()
        
        output_dir = os.path.join("/media",os.getlogin(),usb,"data") #for Linux
        # output_dir = os.path.join("/Volumes",usb,"data") #for Mac
    

    #creating folders if they do not exist
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    if os.path.exists(os.path.join(output_dir,video_folder)) is False:
        v_path = os.path.join(output_dir,video_folder)
        os.mkdir(v_path) 
    if os.path.exists(os.path.join(output_dir,image_folder)) is False:
        i_path = os.path.join(output_dir,image_folder)
        os.mkdir(i_path)

    v_path = os.path.join(output_dir,video_folder)
    i_path = os.path.join(output_dir,image_folder)

    return v_path, i_path


def run(video=False, image=False, both=False, fps=30, delay=1, camera_idx=0, usb='None'):

    #get directories
    video_path, image_path = create_dir(usb)

    if both:
        video = True
        image = True
    
    # Create a VideoCapture object
    logger.info("Starting video stream...")
    cap = cv2.VideoCapture(camera_idx)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    # Obtaining frame resolution. The resolution is source dependent.
    # Convert the resolutions from float to integer.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))

    record = True
    start = False
    capture = True

    start_time = time.time() #start time for delay
    while(True):
        
        ret, frame = cap.read()

        if ret == True:

            #Video recording
            if video and record:
                if not start:
                    video_name = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+".avi") #file name with timestamp
                    vid_out = cv2.VideoWriter(os.path.join(video_path,video_name),cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height)) # Create VideoWriter object.The output is saved in the VideoData folder in 'timestamp'.avi file
                    start = True
                # Write the frame into the 'timestamp'.avi file
                vid_out.write(frame)

            #Release video capture when recording stops
            if not record and start:
                start = False
                vid_out.release()

            #Image recording
            if image and capture:
                if time.time()-start_time >= delay:
                    image_name = "{}".format(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")+".jpeg")
                    cv2.imwrite(os.path.join(image_path,image_name), frame)
                    logger.info(f'Image saved to {os.path.join(image_path,image_name)}')
                    start_time = time.time()

            # Display the resulting frame
            cv2.imshow('frame',frame)
            
            # Press r on keyboard to start/stop image capture
            if cv2.waitKey(1) & 0xFF == ord('c'):
                capture = not capture
                logger.info(f'Image Capture with {delay}s delay in Progress' if capture else f'Image Capture Ended')
                time.sleep(1)
            
            # Press r on keyboard to start/stop video capture
            if cv2.waitKey(1) & 0xFF == ord('r'):
                record = not record
                logger.info(f'Recording in Progress' if record else f'Recording saved to {os.path.join(video_path,video_name)}')
            
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
    parser.add_argument("--video", action='store_true', help="video capture")
    parser.add_argument("--image", action='store_true', help="image capture")
    parser.add_argument("--both", action='store_true', help="image and video capture")
    parser.add_argument("--fps", type=int, default=30, help="video fps")
    parser.add_argument("--delay", type=int, default=1, help="image capture delay")
    parser.add_argument('--camera_idx', type=int, default=0, help='Index of which video source to use')
    parser.add_argument("--usb", type=str, default='None', help="USB stick name")
    args = parser.parse_args()
    logger.info(f'Arguements: ' + ', '.join(f'{k}={v}' for k, v in vars(args).items()))
    return args

def main(args):
    run(**vars(args))

if __name__ == '__main__':
    args = parse_args()
    main(args)