# simple_classifier_py
## Introduction
This application runs an inference on an image using [GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).  Although the sample uses GoogLeNet as the default network, other classifier models can also be used (see [Options](#options-for-run.py) section). The provided Makefile does the following

1. Downloads the Caffe prototxt file and makes few changes necessary to work with the Intel® Neural Compute Stick (NCS1/NCS2) and the OpenVINO toolkit (tested with version 2019 R2.
2. Downloads and generates the required ilsvrc12 data.
3. Downloads the caffemodel weights file from the [Open Model Zoo](https://github.com/opencv/open_model_zoo).
3. Compiles the model to an IR (Intermediate Representation) format file using the Model Optimizer. An IR is a static representation of the model that is compatitible with the OpenVINO Inference Engine API. 
4. There is a run.py provided that does a single inference on a provided image as an example on how to use the network using the OpenVINO Inference Engine Python API.

## Running the Example
To run the example code do the following :
1. Open a terminal and change directory to the sample base directory
2. Type the following command in the terminal: ```make run``` 

# Prerequisites
All development and testing has been done on Ubuntu 16.04 on an x86-64 machine.

This program requires:
- 1 Intel NCS device
- OpenVINO 2019 R2 toolkit

## Makefile
Provided Makefile has various targets that help with the above mentioned tasks.

### make run
Runs the sample application.

### make help
Shows available targets.

### make all
Builds and/or gathers all the required files needed to run the application.

### make data
Gathers all of the required data need to run the sample.

### make deps
Builds all of the dependencies needed to run the sample.

### make install_reqs
Checks required packages that aren't installed as part of the OpenVINO installation. 
 
### make clean
Removes all the temporary files that are created by the Makefile.

# Options for run.py

| Option | Description |
|--------|-------------|
| --ir IR_FILE | Absolute path to the neural network IR file. The default is: ../../caffe/SqueezeNet/squeezenet_v1.0.xml. |
| -l LABEL_FILE, --labels LABEL_FILE | Absolute path to labels file. The default is: ../../data/ilsvrc12/synset_words.txt. |
| -m NUMPY_MEAN_FILE, --mean NUMPY_MEAN_FILE | Network Numpy mean file. The default is: None. |
| -i IMAGE_FILE, --image IMAGE_FILE | Absolute path to the image file. The default is ../../data/images/nps_electric_guitar.png. |
| -n NUMBER_OF_RESULTS_TO_DISPLAY, --number_top NUMBER_OF_RESULTS_TO_DISPLAY | The number of inference results to display. The default is 1. |


