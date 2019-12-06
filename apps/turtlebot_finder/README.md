# Turtlebot Finder
## Introduction
This project uses the Caffe SSD Mobilenet object detection model (https://github.com/chuanqi305/MobileNet-SSD), the Neural Compute Stick, the NCSDK, and the Turtlebot 2 to implement an object finding robot. 

The provided Makefile does the following:
1. Downloads network prototxt and caffemodel.
2. Compiles the network files into a graph file for the NCS device.
 

## Prerequisites
This program requires:
- 1 NCS device
- NCSDK 2.08 or greater
- OpenCV
- ROS Kinetic
- Turtlebot
- Camera (webcam or USB)

## How to run this sample code
1. Run the tb_start.sh script using the command: ***./tb_start.sh*** <br>
You must use the tb_start.sh script to start a ROS Master first before running run.py. The ROS Master must be running in a terminal at all times for communication to exist between the turtlebot and the python script. 

2. Download and compile the network using the command: ***make compile*** <br>
If you do not have a compatible Movidius graph file, use this command to download and compile a graph file for use with the application.

3. Start the application using the command: ***python2 run.py*** or ***make run***<br>
You can also use some built in flags to enable more features and options. <br>
**EXAMPLE COMMAND:** ***python2 run.py -a -f person -c 1*** <br>

#### FLAGS/OPTIONS

* -f, --find
> - default = dog<br>
> AVAILBLE OPTIONS: aeroplane, bicycle, bird, boat, bottle,
> bus, car, cat, chair, cow, diningtable, dog, horse, motorbike,
> person, pottedplant, sheep, sofa, train, tvmonitor<br>
> EXAMPLE: -f person

* -v, --velocity
> - default = 70<br>
> VELOCITY VALUE: integer value 1-100. default: 80 <br>
> EXAMPLE: -v 70

* -c, --cam_src
> - default = 0<br>
> CAMERA INDEX: index of video camera. default: 0<br>
> EXAMPLE: -c 1

* -g, --graph
> - default = graph<br>
> MOVIDIUS GRAPH FILE: compiled graph file. default: graph<br>
> EXAMPLE: -g ssd_mobilenet.graph

* -r, --res
> - default = [1280, 720]<br>
> CAMERA CAPTURE RESOLUTION. default: 1280x720<br>
> EXAMPLE: -r [640, 360]

* -d, --detection 
> - default = False<br>
> Displays the detection window. default: False<br>
> EXAMPLE: -d

* -a, --animation 
> - default = False<br>
> Displays the animation window. default: False<br>
> EXAMPLE: -a

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application except the ncsdk.  This must be done as a separate step.

### make run
Runs the provided python program which runs inferences based on images from a webcam/USB webcam.

### make clean
Removes all the temporary files that are created by the Makefile

## Note
- Make sure the ROS Master is started before the application starts. Do this by running the tb_start.sh script then running run.py.  
- Make sure the velocity is 40 or higher or else the turtlebot may move too slow. 
- This application requires python 2.
