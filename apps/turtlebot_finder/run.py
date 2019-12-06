#! /usr/bin/env python2

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

import numpy
import cv2
import sys
import argparse
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from robot import Robot
from openvino.inference_engine import IENetwork, IECore

ir = "./mobilenet-ssd.xml"
IDLE_VIDEO = "animations/robot_idle.gif"        # video for animation at idle
DETECTED_VIDEO = "animations/robot_happy.gif"   # video for animation when detecting an object

NETWORK_WIDTH = 300                     # model width
NETWORK_HEIGHT = 300                    # model height
dim = (NETWORK_WIDTH, NETWORK_HEIGHT)   # dimensions for model input

MAX_VELOCITY = 1.0                  # maximum velocity in rads/sec. default: 1.0
tb_vel = 0.0                        # actual velocity of the robot
WINDOW_WIDTH = 1280                 # window size width. default: 1280
WINDOW_HEIGHT = 720                 # window size height. default: 720
# other 16:9 resolutions for reference: 640x360, 768x432, 896x504, 1280x720, 1920x1080

RETURN_HOME_DELAY = 2.0              # delay for moving back to home position
NORMAL_CMD_DELAY = 0.00              # delay for sending robot commands
INFERENCE_DELAY = 0.00

video_window_name = "Turtlebot Finder - Press q to quit"
cam_window_name = "Turtlebot Finder (Detection) - Press q to quit"
        
ARGS = None
MIN_OBJ_SCORE = 40
device = "MYRIAD"


# ***************************************************************
# Labels for the classifications for the network.
# ***************************************************************

LABELS = ('background',
          'aeroplane',   'bicycle', 'bird',  'boat',      'bottle',
          'bus',         'car',     'cat',   'chair',     'cow', 
          'diningtable', 'dog',     'horse', 'motorbike', 'person', 
          'pottedplant', 'sheep',   'sofa',  'train',     'tvmonitor')

# mask used for filtering out other objects. default: dogs only
LABELS_MASK = [0,
           0,             0,         0,       0,           0, 
           0,             0,         0,       0,           0, 
           0,             0,         0,       0,           0, 
           0,             0,         0,       0,           0]


def arg_parse():
    global ARGS, tb_vel
    parser = argparse.ArgumentParser(
                         description="Turtlebot Object Finder using the Intel Movidius Neural Compute Stick")
    
    parser.add_argument('-f', '--find', 
                        action = 'store', nargs = '+', 
                        type = str, metavar = "<OBJECT>",
                        default = 'dog',
                        help = "AVAILBLE OPTIONS: aeroplane, bicycle, bird, boat, bottle,\
                        bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor. \n default: dog")

    parser.add_argument('-v', '--velocity', 
                        type = float, metavar = "<ROBOT VELOCITY>",
                        default = 65,
                        help = "VELOCITY VALUE: integer value 1-100. default: 65" )
                             
    parser.add_argument('-c', '--cam_src', 
                        type = int, metavar = "<CAMERA INDEX>",
                        default = 0,
                        help = "CAMERA INDEX: index of video camera. default: 0" )

    parser.add_argument('-g', '--graph', 
                        type = str, metavar = "<MOVIDIUS GRAPH FILE>",
                        default = 'graph',
                        help = "MOVIDIUS GRAPH FILE: compiled graph file. default: graph" )
       
    parser.add_argument('-r', '--res', 
                        action = 'store', nargs = '+',
                        type = int, metavar = "<CAMERA CAPTURE RESOLUTION>",
                        default = [1280, 720],
                        help = "CAMERA CAPTURE RESOLUTION. default: 1280x720" )
   
    parser.add_argument('-d', '--detection', 
                        action = 'store_const', const = True,
                        default = False,
                        help = "Displays the detection window. default: False" )

    parser.add_argument('-a', '--animation', 
                        action = 'store_const', const = True,
                        default = False,
                        help = "Displays the animation window. default: False" )
 
    ARGS = parser.parse_args()
    
    # determine actual robot velocity based on user input
    tb_vel = ARGS.velocity / 100.0
    if (tb_vel > MAX_VELOCITY or tb_vel <= 0.0):
        print("Velocity error. Please enter a value between 1-100")
        exit()
        
    # mask all object that we are not looking for
    for i in range(len(LABELS)):
        if(LABELS[i] in ARGS.find):
            LABELS_MASK[i] = 1;

    if (1 not in LABELS_MASK):
        print("Error: No target objects specified.")
        exit()


def display_settings():
    print
    print(" --------------- Turtlebot Object Finder -----------------")
    arg_parse()
    
    print("   * Camera source index ...................... " + str(ARGS.cam_src))
    print("   * Camera capture resolution ................ " + str(ARGS.res[0]) + "x" + str(ARGS.res[1]))
    print("   * Turtlebot velocity ....................... " + str(ARGS.velocity))
    print("   * Movidius graph file ...................... " + str(ARGS.graph))
    print("   * Animation window ......................... " + str(ARGS.animation))
    print("   * Detection window ......................... " + str(ARGS.detection))
    print("   * Objects to look for:")
    for i in range(len(LABELS)):
        if (LABELS_MASK[i] != 0):
            print("       - " + LABELS[i] + " ")
    print(" ---------------------------------------------------------")
    print
    print("Starting...")
    if (ARGS.animation == False and ARGS.detection == False):
        print("Press CTRL-C to stop robot and exit program.")


def setup_cv_windows():
    global cam_window_name, video_window_name
    if (ARGS.detection == True):
        # Opencv window initialization
        cv2.namedWindow(cam_window_name, 16)
        cv2.resizeWindow(cam_window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    if (ARGS.animation == True):
        cv2.namedWindow(video_window_name, 16)
        cv2.resizeWindow(video_window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

def setup_capture_objs():    
    # set up capture object for the camera
    camera_capture = cv2.VideoCapture(ARGS.cam_src)
    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, ARGS.res[0])
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, ARGS.res[1])
    
    return camera_capture
    



def preprocess_image(src_image, n, c, h, w):
    """ Perform image preprocessing: First resize the image, then subtract mean and divide by std dev. """
    image_copy = src_image
    image_copy = cv2.resize(image_copy, (w, h))
    image_copy = numpy.transpose(image_copy, (2, 0, 1))
    preprocessed_img = image_copy.reshape((n, c, h, w))
    
    return preprocessed_img


def text_setup(frame, labels_list, class_id, confidence, box_left, box_top):
    # label shape and colorization for displaying
    label_text = labels_list[class_id] + " " + str("{0:.2f}".format(confidence))
    label_background_color = (70, 120, 70) # grayish green background for text
    label_text_color = (255, 255, 255)   # white text
    
    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = int(box_left)
    label_top = int(box_top) - label_size[1]
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]

    # set up the greenish colored rectangle background for text
    cv2.rectangle(frame, (label_left - 1, label_top - 5),(label_right + 1, label_bottom + 1), label_background_color, -1)
    # set up text
    cv2.putText(frame, label_text, (int(box_left), int(box_top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


def run_inference(exec_net, image_to_classify, cam_image, input_blob, output_blob):
    """ Run an inference on the NCS, then determine if it we need to draw a box around the object. """
    # send the image for preprocessing """
    
    cur_request_id = 0
    image_w = ARGS.res[0]
    image_h = ARGS.res[1]
    box_color = (0, 255, 0)
    box_thickness = 1
    
    exec_net.start_async(request_id=cur_request_id, inputs={input_blob: image_to_classify})
            
    # wait for inference to complete
    if exec_net.requests[cur_request_id].wait(-1) == 0:
        # get the inference result
        inference_results = exec_net.requests[cur_request_id].outputs[output_blob]
        # process the results
        for num, detection_result in enumerate(inference_results[0][0]):
            percentage = int(detection_result[2] * 100.0)
            # only draw bounding boxes around the first detected valid object.    
            if (LABELS_MASK[int(detection_result[1])] != 0 and percentage >= MIN_OBJ_SCORE):
                
                # find the left and right sides of the bounding box
                box_left = int(detection_result[3] * image_w)
                box_top = int(detection_result[4] * image_h)
                box_right = int(detection_result[5] * image_w)
                box_bottom = int(detection_result[6] * image_h)
                class_id = int(detection_result[1])
                # set up the text to display with the bounding box in the frame
                text_setup(cam_image, LABELS, class_id, detection_result[2], box_left, box_top)
                # set up the detection box in the frame
                cv2.rectangle(cam_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

                # return the center of the detected object
                return ((box_right - box_left)/2) + box_left

    # we use a -1 to symbolize that no valid objects were found
    return -1




# This function is called from the entry point to do
# all the work of the program
def main():
    global NORMAL_CMD_DELAY, RETURN_HOME_DELAY
    current_cmd_delay = NORMAL_CMD_DELAY
    current_obj = False
    prev_obj = False
    
    frame_counter = 0
    delay_counter = 0
    frame_limit = 1
    
    # Display the app settings
    display_settings()
    # Set up the opencv windows
    setup_cv_windows()
    # Set up the capture objects
    camera_capture = setup_capture_objs()

    # Robot initialization
    turtle_bot = Robot(tb_vel, ARGS.res)
    # set the robot home position
    turtle_bot.set_home_pos()
    # start timer for robot command interval
    cmd_start_time = time.time()
    inference_start_time = time.time()
    obj_x_coord = -1
    
    ####################### 1. Setup Plugin and Network #######################
    # Set up the inference engine core and load the IR files
    ie = IECore()
    net = IENetwork(model = ir, weights = ir[:-3] + 'bin')
    # Get the input and output node names
    input_blob = next(iter(net.inputs))
    output_blob = next(iter(net.outputs))
    
    # Get the input and output shapes from the input/output nodes
    input_shape = net.inputs[input_blob].shape
    output_shape = net.outputs[output_blob].shape
    n, c, h, w = input_shape
    x, y, detections_count, detections_size = output_shape
    exec_net = ie.load_network(network = net, device_name = device)
    # main loop for robot object finder
    while (True):
        # read an image from the camera, exit if unable to read from the camera
        ret_val, cam_image = camera_capture.read()
        if (ret_val != True):
            print("Error: Can't read frame from camera!")
            break
                
        # flip the image horizontally for more natural orientation (mirror)
        cam_image = cv2.flip(cam_image, 1)
        
        inference_end_time = time.time()
        inference_elapsed_time = inference_end_time - inference_start_time
        
        if (inference_elapsed_time > INFERENCE_DELAY):
        # make an inference and get the object's location
            image_to_classify = preprocess_image(cam_image, n, c, h, w)
            obj_x_coord = run_inference(exec_net, image_to_classify, cam_image, input_blob, output_blob)
            inference_start_time = time.time()
        
        # update the history of objects detected
        prev_obj = current_obj
        if (obj_x_coord == -1):
            current_obj = False
        else:
            current_obj = True
        
        # create a 2 second pause from sending robot any commands if
        # an obj was previously detected but no objs are present
        if (current_obj == False and prev_obj == True and current_cmd_delay == NORMAL_CMD_DELAY):
            current_cmd_delay = RETURN_HOME_DELAY
            cmd_start_time = time.time()
        
        # update the timer
        cmd_end_time = time.time()
        cmd_elapsed_time = cmd_end_time - cmd_start_time    
        
        # if it's time to move the bot, determine what it should do
        if (cmd_elapsed_time > current_cmd_delay):
            # determine what the robot should do based on where the object is in the image frame        
            turtle_bot.determine_robot_action(obj_x_coord)
            turtle_bot.take_robot_action()
            # reset the bot command transmission timer
            cmd_start_time = time.time()
            # remove pause
            current_cmd_delay = NORMAL_CMD_DELAY
            
        ######################### Animation code ###########################
        # TO DO: needs work
        # loops animations in specific frame "limits". 
        # ex: loop only from frame 1 to frame 5 until 50 frames have passed, then
        #     play the entire animation from frame 1 to frame 12. repeat. 
        # 
        # 
        # frame counter keeps track of the number of frames that have passed
        # delay counter keeps track of the number of frames before playing the entire animation
        
        if (ARGS.animation == True):

            frame_counter += 1
            delay_counter += 1
            
            #print("fc: " + str(frame_counter) + " bc: " + str(delay_counter))
            #### Idle animation
            if (obj_x_coord == -1 or turtle_bot.get_vel() != 0):
                # This code makes the gif loop correctly else the animations just blink repeatedly
                # decide if it is time to play entire animation
                if (delay_counter >= 50):
                    frame_limit = 12        # play the entire aniimation 
                    delay_counter = 0
                
                # show certain number of frames depending on the delay counter
                if (frame_counter >= frame_limit):
                    video_capture = cv2.VideoCapture(IDLE_VIDEO)
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                    frame_limit = 1         # only play the first frame
                    frame_counter = 0
            #### Detected an object animation 
            elif (obj_x_coord > 0 and turtle_bot.get_vel() == 0):
                # decide if it is time to play the entire animation
                if (delay_counter >= 50):
                    frame_limit = 12        # play entire animation
                    delay_counter = 0
                                        
                if (frame_counter >= frame_limit):
                    video_capture = cv2.VideoCapture(DETECTED_VIDEO)
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_limit = 5         # play only part of the animation
                    frame_counter = 0
                    
            # Read an image from the animation video
            ret_val, video_image = video_capture.read()
            if (ret_val != True):
                print("Error: Can't read frame from video file!")
                break
                
            # display the video frame in the video window
            cv2.imshow(video_window_name, video_image)
        #####################################################################
        
        # show camera frames in camera window
        if (ARGS.detection == True):
            cv2.imshow(cam_window_name, cam_image)
        
        if (cv2.waitKey( 1 ) & 0xFF == ord( 'q' )):
            break
    
    
    # Release the camera stream
    camera_capture.release()
    # Release the animated stream
    if (ARGS.animation == True):
        video_capture.release()    
        
    print("Finished.")


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())

