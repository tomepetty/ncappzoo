#! /usr/bin/env python2

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.
from __future__ import division
import numpy
import cv2
import sys
import argparse
import time
import os
from openvino.inference_engine import IENetwork, IECore

from Robot import Robot
from Gif_animation_handler import Gif_animation_handler

device = "MYRIAD"

MAX_TB_VELOCITY = 1.0               # maximum velocity in rads/sec. default: 1.0
WINDOW_WIDTH = 1280                 # window size width. default: 1280
WINDOW_HEIGHT = 720                 # window size height. default: 720
# other 16:9 resolutions for reference: 640x360, 768x432, 896x504, 1280x720, 1920x1080

RETURN_HOME_DELAY = 2.0              # delay for moving back to home position
NORMAL_CMD_DELAY = 0.00              # delay for sending robot commands
INFERENCE_DELAY = 0.00

video_window_name = "Turtlebot Finder - Press q to quit"
cam_window_name = "Turtlebot Finder (Detection) - Press q to quit"
        
ARGS = None

# Adjust these thresholds
DETECTION_THRESHOLD = 0.70

# Used for display
BOX_COLOR = (0,255,0)
LABEL_BG_COLOR = (70, 120, 70) # greyish green background for text
TEXT_COLOR = (255, 255, 255)   # white text
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX


def arg_parse():
    global ARGS
    parser = argparse.ArgumentParser(
                         description="Turtlebot Object Finder using the Intel Neural Compute Stick 2")
    
    parser.add_argument('-f', '--find', 
                        action = 'store', nargs = '+', 
                        type = str, metavar = "<OBJECT>",
                        default = 'person',
                        help = "DEFULT AVAILABLE OPTIONS (SSD Mobilenet v1 caffe): aeroplane, \
                        bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, \
                        dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor. \
                        \n default: person")

    parser.add_argument('-v', '--velocity', 
                        type = float, metavar = "<ROBOT VELOCITY>",
                        default = 65,
                        help = "VELOCITY VALUE: integer value 1-100. default: 65" )
                             
    parser.add_argument('-c', '--cam_src', 
                        type = int, metavar = "<CAMERA INDEX>",
                        default = 0,
                        help = "CAMERA INDEX: index of video camera. default: 0" )

    parser.add_argument('-x', '--xml', 
                        type = str, metavar = "<OPENVINO IR XML FILE>",
                        default = '../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.xml',
                        help = "OPENVINO IR XML FILE. \n default: ../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.xml" )
                    
    parser.add_argument('-b', '--bin', 
                        type = str, metavar = "<OPENVINO IR BIN FILE>",
                        default = '../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.bin',
                        help = "OPENVINO IR BIN FILE. \n default: ../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.bin" )
                            
    parser.add_argument('-l', '--labels', 
                        type = str, metavar = "<LABELS FILE PATH>",
                        default = '../../networks/ssd_mobilenet_v1_caffe/labels.txt',
                        help = "LABELS FILE PATH: text file with all labels/categories for the model. \n default: ../../networks/ssd_mobilenet_v1_caffe/labels.txt" )

    parser.add_argument('--class_name', 
                        type = str, metavar = "<NAME OF THE OBJECT DETECTOR CLASS>",
                        default = 'Ssd_mobilenet_object_detector',
                        help = "NAME OF THE OBJECT DETECTOR PROCESSOR CLASS. \n default: Ssd_mobilenet_object_detector" )
                        
    parser.add_argument('-r', '--cam_res', 
                        action = 'store', nargs = '+',
                        type = int, metavar = "<CAMERA CAPTURE RESOLUTION>",
                        default = [640, 480],
                        help = "CAMERA CAPTURE RESOLUTION. default: 640x480" )
   
    parser.add_argument('-d', '--detection_window', 
                        action = 'store_const', const = True,
                        default = False,
                        help = "Displays the detection window. default: False" )

    parser.add_argument('-a', '--animation_window', 
                        action = 'store_const', const = True,
                        default = False,
                        help = "Displays the animation window. default: False" )
 
    ARGS = parser.parse_args()


def set_tb_velocity():
    # Determine robot rotational velocity based on user input
    tb_rotation_velocity = ARGS.velocity / 100.0
    if (tb_rotation_velocity > MAX_TB_VELOCITY or tb_rotation_velocity <= 0.0):
        print("Velocity error. Please enter a value between 1-100")
        exit()
    return tb_rotation_velocity


def read_and_mask_labels():    
    # Read labels 
    with open(ARGS.labels) as labels_file:
        label_list = labels_file.read().splitlines()
    
    label_mask = [0] * len(label_list)
    
    # Mask all object that we are not looking for
    for i in range(len(label_list)):
        if(label_list[i] in ARGS.find):
            label_mask[i] = 1;
            
    # Exit if target labels do not match network labels
    if (1 not in label_mask):
        print("Error: No target objects specified or target label does not match network labels.")
        exit()
    return label_list, label_mask
    

def display_settings_to_console(label_list, label_mask):
    print
    print(" --------------- Turtlebot Object Finder -----------------")
    
    print("   * Camera source index ...................... " + str(ARGS.cam_src))
    print("   * Camera capture resolution ................ " + str(ARGS.cam_res[0]) + "x" + str(ARGS.cam_res[1]))
    print("   * Turtlebot velocity ....................... " + str(ARGS.velocity))
    print("   * OpenVINO graph file ...................... " + str(ARGS.model))
    print("   * Animation window ......................... " + str(ARGS.animation_window))
    print("   * Detection window ......................... " + str(ARGS.detection_window))
    print("   * Objects to look for:")

    for i in range(len(label_list)):
        if (label_mask[i] != 0):
            print("       - " + label_list[i] + " ")
    print(" ---------------------------------------------------------")
    print
    print("Starting...")
    if (ARGS.animation_window == False and ARGS.detection_window == False):
        print("Press CTRL-C to stop robot and exit program.")


def setup_cv_windows():
    if (ARGS.detection_window == True):
        # Opencv window initialization
        cv2.namedWindow(cam_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cam_window_name, WINDOW_WIDTH, WINDOW_HEIGHT)


def setup_capture_objs(res):    
    # set up capture object for the camera
    camera_capture = cv2.VideoCapture(ARGS.cam_src)
    camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(res[0]))
    camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(res[1]))
    return camera_capture
    

def get_object_x_coordinate(cam_image, detected_objs, label_list, label_mask):
    # return the center horizontal value for the largest object detected (by area of bounding box)
    obj_x_coord = -1    # -1 denotes that no target objects were found
    largest_bbox_area = 0
    if detected_objs == None:
        print "None type"
        return obj_x_coord
        
    if len(detected_objs) > 0: 
        for object_index in range(len(detected_objs)):
            xmin = int(detected_objs[object_index][0])
            ymin = int(detected_objs[object_index][1])
            xmax = int(detected_objs[object_index][2])
            ymax = int(detected_objs[object_index][3])
            confidence = detected_objs[object_index][4]
            class_id = detected_objs[object_index][5]
            # Draw rectangles for all target objects (aka not masked objects)
            if label_mask[class_id] != 0:
                # Set up the text for display
                cv2.rectangle(cam_image, (xmin, ymin), (xmax, ymin+20), LABEL_BG_COLOR, -1)
                cv2.putText(cam_image, label_list[class_id] + ': %.2f' % confidence, (xmin+5, ymin+15), TEXT_FONT, 0.5, TEXT_COLOR, 1)
                # Set up the bounding box
                cv2.rectangle(cam_image, (xmin, ymin), (xmax, ymax), BOX_COLOR, 1)
                
                bbox_area = (xmax - xmin) * (ymax -ymin)
                if bbox_area > largest_bbox_area:
                    largest_bbox_area = bbox_area
                    obj_x_coord = ((xmax - xmin)/2) + xmin 
    return obj_x_coord


# This function is called from the entry point to do
# all the work of the program
def main():
    # Set some default values
    current_cmd_delay = NORMAL_CMD_DELAY    # A time delay before sending the robot a command
    current_obj = False         # Boolean. Detecting an object in the current frame
    prev_obj = False            # Boolean. Having detected a previous object in the previous frame
    obj_x_coord = -1            # Default x coordinate value. -1 denotes no object was detected

    # Parse command line arguments and set IR path
    arg_parse()
    xml_path = ARGS.model
    bin_path = ARGS.bin
    
    # Set up windows and capture objects
    setup_cv_windows()
    camera_capture = setup_capture_objs(ARGS.cam_res)

    # Import object detector class
    model_path = ARGS.model.replace(os.path.basename(ARGS.model), "")
    sys.path.append(model_path)
    class_name = getattr(__import__(ARGS.class_name, fromlist=[ARGS.class_name]), ARGS.class_name)
    
    # Read and set the target labels then display all app settings to console
    labels_path = ARGS.labels
    label_list, label_mask = read_and_mask_labels()
    display_settings_to_console(label_list, label_mask)
    
    # Robot object initialization
    tb_rotation_velocity = set_tb_velocity()
    #turtle_bot = Robot(tb_rotation_velocity, ARGS.cam_res)
    #turtle_bot.set_home_pos()
    
    # Create animation handler object
    #animation_handler = Gif_animation_handler(video_window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Start various timers
    cmd_start_time = time.time()
    inference_start_time = time.time()
    
    ####################### Setup Plugin and Network #######################
    # Set up the inference engine core object
    ie = IECore()
    net = IENetwork(model = xml_path, weights = bin_path)
    # create the network processor object
    network_processor = class_name(ie, net, device)
    network_processor.set_parameter("detection_threshold", 0.70)
    
    # main loop for robot object finder
    while (True):
        # Read an image from the camera, exit if unable to read from the camera
        ret_val, cam_image = camera_capture.read()
        if (ret_val != True):
            print("Error: Can't read frame from camera!")
            break
                
        # Flip the image horizontally for more natural orientation (mirrored)
        cam_image = cv2.flip(cam_image, 1)
        
        ####################### Preprocessing, Inference and Postprocesing #######################
        # Check to see if its time to perform inference
        inference_end_time = time.time()
        inference_elapsed_time = inference_end_time - inference_start_time
        if (inference_elapsed_time > INFERENCE_DELAY):
            # Preprocess the image, make the inference
            #image_to_classify = network_processor.preprocess_image(cam_image)
            detected_objs = network_processor.run_inference_sync(cam_image)
            # Get the center x-coordinate of the object
            obj_x_coord = get_object_x_coordinate(cam_image, detected_objs, label_list, label_mask)
            # Reset inference timer
            inference_start_time = time.time()
        
        # Update the history of objects detected
        prev_obj = current_obj
        if (obj_x_coord == -1):
            current_obj = False
        else:
            current_obj = True
        
        ####################### Handling the robot #######################
        # The app will not send the robot commands for 2 seconds if
        # an object was previously detected but no objects are present in the current frame.
        if (current_obj == False and prev_obj == True and current_cmd_delay == NORMAL_CMD_DELAY):
            current_cmd_delay = RETURN_HOME_DELAY
            cmd_start_time = time.time()
        
        # Update the turtlebot command timer and check elapsed time
        cmd_end_time = time.time()
        cmd_elapsed_time = cmd_end_time - cmd_start_time    
        
        # If it's time to give the robot a command, determine what it should do
        if (cmd_elapsed_time > current_cmd_delay):
            # Determine what the robot should do based on where the object is in the image frame        
            #turtle_bot.determine_robot_action(obj_x_coord)
            #turtle_bot.take_robot_action()
            # Reset the robot command timer
            cmd_start_time = time.time()
            # Ensure the command timer is back to normal
            current_cmd_delay = NORMAL_CMD_DELAY
     
        ####################### Handling the animations #######################
        if (ARGS.animation_window == True): 
            animation_handler.determine_animations(turtle_bot, obj_x_coord)
        
        # Show camera frames in camera window
        if (ARGS.detection_window == True):
            cv2.imshow(cam_window_name, cam_image)
        
        if (cv2.waitKey(1) & 0xFF == ord( 'q' )):
            break
    
    # Release the camera stream
    camera_capture.release()
    print("Finished.")


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())

