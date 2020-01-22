# Turtlebot Finder
## Introduction
This project uses the Caffe SSD Mobilenet object detection model (https://github.com/chuanqi305/MobileNet-SSD), the Neural Compute Stick 2, OpenVINO, and the Turtlebot 2 to implement an object finding robot. 

The provided Makefile does the following:
1. Download the compile the SSD Mobilenet network to IR
2. Run the app

## Prerequisites
This program requires:
- 1 NCS/NCS2 device
- OpenVINO R3
- ROS Kinetic
- Turtlebot 2
- Camera (webcam or USB)
- Python 2.7

## How to install ROS and Turtlebot with Kobuki Base
1. Install ROS kinetic - **http://wiki.ros.org/kinetic/Installation/Ubuntu**
2. Install Turtlebot for ROS kinetic - **sudo apt-get install ros-kinetic-turtlebot ros-kinetic-turtlebot-apps ros-kinetic-turtlebot-interactions ros-kinetic-turtlebot-simulator ros-kinetic-kobuki-ftdi ros-kinetic-ar-track-alvar-msgs**
3. Setup Kobuki base - **. /opt/ros/kinetic/setup.bash** 
4. Create the udev rules for the Kobuki base - **rosrun kobuki_ftdi create_udev_rules**

## How to run this sample code
1. Open a terminal and navigate to the Turtlebot_finder folder. Run the tb_start.sh script using the command: **./tb_start.sh** <br>

You must use the tb_start.sh script to start a ROS Master first before running run.py. The ROS Master must be running in a terminal at all times for communication to exist between the turtlebot and the python script. 

2. Open another terminal and navigate to the Turtlebot_finder folder. Set your OpenVINO environment variables for the terminal by sourcing your setupvars.bin file. This file should be located in the bin folder of your OpenVINO install location. <br>
EXAMPLE: **source ~/intel/OpenVINO/bin/setupvars.sh** <br>

3. Add the OpenVINO Python2 folder to your Python Path environment variable. <br>
EXAMPLE: **export PYTHONPATH=/home/YOUR_USER_NAME/intel/openvino_2019.3.376/python/python2.7:$PYTHONPATH**

4. Start the application using the command: ***python2 run.py*** or ***make run***<br>
You can also use some built in flags to enable more features and options. <br>
EXAMPLE: **python2 run.py -a -f person** <br>

#### FLAGS/OPTIONS

* -f, --find
> * default = dog<br>
> * AVAILABLE OPTIONS: aeroplane, bicycle, bird, boat, bottle,
> bus, car, cat, chair, cow, diningtable, dog, horse, motorbike,
> person, pottedplant, sheep, sofa, train, tvmonitor<br>
> * EXAMPLE: -f=person

* -v, --velocity
> * default = 70<br>
> * Velocity value: integer value 1-100. <br>
> * EXAMPLE: -v=70

* -c, --cam_src
> * default = 0<br>
> * Camera index: index of video camera.<br>
> * EXAMPLE: -c=0

* -x, --xml
> * default = ../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.xml<br>
> * OpenVINO IR xml file. <br>
> * EXAMPLE: -x=../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.xml

* -b, --bin
> * default = ../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.bin<br>
> * OpenVINO IR bin file. <br>
> * EXAMPLE: -b=../../networks/ssd_mobilenet_v1_caffe/mobilenet-ssd.bin

* -l, --labels
> * default = ../../networks/ssd_mobilenet_v1_caffe/labels.txt<br>
> * Model labels text file. <br>
> * EXAMPLE: -x=../../networks/ssd_mobilenet_v1_caffe/labels.txt

* --class_name
> * default = Ssd_mobilenet_object_detector<br>
> * Object detector class name inherited from Object_detector (found in shared/Python). <br>
> * EXAMPLE: -x=Ssd_mobilenet_object_detector

* -r, --cam_res
> * default = [1280, 720]<br>
> * Camera capture resolution.<br>
> * EXAMPLE: -r=[640, 360]

* -d, --detection_window
> * default = False<br>
> * Displays the detection window.<br>
> * EXAMPLE: -d

* -a, --animation_window
> * default = False<br>
> * Displays the animation window.<br>
> * EXAMPLE: -a

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make default_model
Compiles an IR file from a default model to be used when running the sample.

### make install-reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 

### make uninstall-reqs
Uninstalls requirements that were installed by the sample program.
 
### make clean
Removes all the temporary files that are created by the Makefile.

## Note
- Make sure the ROS Master is started before the application starts. Do this by running the tb_start.sh script then running run.py.  
- Make sure the velocity is 40 or higher or else the turtlebot may move too slow. 
- This application requires Python 2.
