#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.

GREEN = '\033[1;32m'
RED = '\033[1;31m'
NOCOLOR = '\033[0m'
YELLOW = '\033[1;33m'
DEVICE = "MYRIAD"

try:
    from openvino.inference_engine import IENetwork, IECore
except:
    print(RED + '\nPlease make sure your OpenVINO environment variables are set by sourcing the' + YELLOW + ' setupvars.sh ' + RED + 'script found in <your OpenVINO install location>/bin/ folder.\n' + NOCOLOR)
    exit(1)


import sys
import numpy
import cv2
import time
import csv
import os
from sys import argv
from ssd_mobilenet_processor import ssd_mobilenet_processor
    
    
# name of the opencv window
cv_window_name = "Video Objects - SSD Mobilenet"
ssd_ir = "mobilenet-ssd.xml"

# labels AKA classes.  The class IDs returned
# are the indices into this list
LABELS_FILE_NAME = 'labels.txt'

# only accept classifications with 1 in the class id index.
# default is to accept all object clasifications.
# for example if object_classifications_mask[1] == 0 then
#    will ignore aeroplanes
object_classifications_mask = [1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1]

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
DEFAULT_INIT_MIN_SCORE = 60
min_score_percent = DEFAULT_INIT_MIN_SCORE

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0

# read video files from this directory
input_video_path = '.'

# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(source_image):
    resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    
    return resized_image

# handles key presses by adjusting global thresholds etc.
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent += 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent -= 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True



#return False if found invalid args or True if processed ok
def handle_args():
    global resize_output, resize_output_width, resize_output_height, min_score_percent, object_classifications_mask
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).lower().startswith('exclude_classes=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                exclude_list = str(val).split(',')
                for exclude_id_str in exclude_list:
                    exclude_id = int(exclude_id_str)
                    if (exclude_id < 0 or exclude_id>len(labels)):
                        print("invalid exclude_classes= parameter")
                        return False
                    print("Excluding class ID " + str(exclude_id) + " : " + labels[exclude_id])
                    object_classifications_mask[int(exclude_id)] = 0
            except:
                print('Error with exclude_classes argument. ')
                return False;

        elif (str(an_arg).lower().startswith('init_min_score=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                init_min_score_str = val
                init_min_score = int(init_min_score_str)
                if (init_min_score < 0 or init_min_score > 100):
                    print('Error with init_min_score argument.  It must be between 0-100')
                    return False
                min_score_percent = init_min_score
                print ('Initial Minimum Score: ' + str(min_score_percent) + ' %')
            except:
                print('Error with init_min_score argument.  It must be between 0-100')
                return False;

        elif (str(an_arg).lower().startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True



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
    
    
# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def process_and_display_results(output, frame, labels_list):
    box_color = (0, 255, 0)
    box_thickness = 1
    # number of boxes returned
    num_valid_boxes = int(len(output)/7)

    for box_index in range(num_valid_boxes):
        # set up the text to display with the bounding box in the frame
        text_setup(frame, labels_list, output[1+box_index*7], output[2+box_index*7], output[3+box_index*7], output[4+box_index*7])
        # set up the detection box in the frame
        box_left = output[3+box_index*7]
        box_top = output[4+box_index*7]
        box_right = output[5+box_index*7]
        box_bottom = output[6+box_index*7]
        cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
        
        
    # display text to let user know how to quit
    cv2.rectangle(frame, (0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(frame, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


# prints usage information
def print_usage():
    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('                  Default isto not resize, use size of video frames.')
    print('  init_min_score - set the minimum score for a box to be recognized')
    print('                  must be a number between 0 and 100 inclusive.')
    print('                  Default is: ' + str(DEFAULT_INIT_MIN_SCORE))

    print('  exclude - comma separated list of object class IDs to exclude from following:')
    index = 0
    for oneLabel in labels:
        print("                 class ID " + str(index) + ": " + oneLabel)
        index += 1
    print('            must be a number between 0 and ' + str(len(labels)) + ' inclusive.')
    print('            Default is to exclude none.')

    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080 init_min_score=50 exclude_classes=5,11')


# This function is called from the entry point to do
# all the work.
def main():
    global resize_output, resize_output_width, resize_output_height

    if (not handle_args()):
        print_usage()
        return 1

    ie = IECore()
    # Create Tiny Yolo and GoogLeNet processors for running inferences. 
    # Please see tiny_yolo_processor.py and googlenet_processor.py for more information.
    ssd_processor = ssd_mobilenet_processor(ssd_ir, ie, DEVICE)


    # get list of all the .mp4 files in the image directory
    input_video_filename_list = os.listdir(input_video_path)
    input_video_filename_list = [i for i in input_video_filename_list if i.endswith('.mp4')]

    if (len(input_video_filename_list) < 1):
        # no images to show
        print('No video (.mp4) files found')
        return 1

    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)
    
    labels_list = numpy.loadtxt(LABELS_FILE_NAME, str, delimiter='\t')
    with open(LABELS_FILE_NAME) as labels_file:
        labels_list = labels_file.read().splitlines()

    exit_app = False
    while (True):
        for input_video_file in input_video_filename_list :

            cap = cv2.VideoCapture(input_video_file)

            actual_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

            if ((cap == None) or (not cap.isOpened())):
                print ('Could not open video device.  Make sure file exists:')
                print ('file name:' + input_video_file)
                print ('Also, if you installed python opencv via pip or pip3 you')
                print ('need to uninstall it and install from source with -D WITH_V4L=ON')
                print ('Use the provided script: install-opencv-from_source.sh')
                exit_app = True
                break

            frame_count = 0
            start_time = time.time()
            end_time = start_time

            while(True):
                ret, display_image = cap.read()

                if (not ret):
                    end_time = time.time()
                    print("No image from from video device, exiting")
                    break

                # check if the window is visible, this means the user hasn't closed
                # the window via the X button
                prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
                if (prop_val < 0.0):
                    end_time = time.time()
                    exit_app = True
                    break

				# do inference
                objects = ssd_processor.ssd_mobilenet_inference(display_image, cap.get(3), cap.get(4))
                # display results
                process_and_display_results(objects, display_image, labels_list)
                
                if (resize_output):
                    display_image = cv2.resize(display_image,
                                               (resize_output_width, resize_output_height),
                                               cv2.INTER_LINEAR)
                cv2.imshow(cv_window_name, display_image)

                raw_key = cv2.waitKey(1)
                if (raw_key != -1):
                    if (handle_keys(raw_key) == False):
                        end_time = time.time()
                        exit_app = True
                        break
                frame_count += 1

            frames_per_second = frame_count / (end_time - start_time)
            print('Frames per Second: ' + str(frames_per_second))

            cap.release()

            if (exit_app):
                break;

        if (exit_app):
            break


    cv2.destroyAllWindows()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
