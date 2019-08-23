#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

class ssd_mobilenet_processor:


    detection_threshold = 0.50
    cur_request_id = 0

    # Initialize the class instance
    # ssd_ir is the ssd intermediate representation file
    # ie is the inference core object
    # device is the name of the plugin/device to use for inference
    # also sets up the network input/output
    def __init__(self, ssd_ir: str, ie: IECore, device: str):
        self._ie = ie
        
        # Create the network object
        self._ssd_net = IENetwork(model = ssd_ir, weights = ssd_ir[:-3] + 'bin')
        # Set up the input and output blobs
        self._ssd_input_blob = next(iter(self._ssd_net.inputs))
        self._ssd_output_blob = next(iter(self._ssd_net.outputs))
        self._ssd_input_shape = self._ssd_net.inputs[self._ssd_input_blob].shape
        self._ssd_output_shape = self._ssd_net.outputs[self._ssd_output_blob].shape
        # Load the network
        self._ssd_exec_net = ie.load_network(network = self._ssd_net, device_name = device)
        # Get the input shape
        self._ssd_n, self._ssd_c, self.ssd_h, self.ssd_w = self._ssd_input_shape

    
    # Performs the image preprocessing and makes an inference
    # Returns a list of detected object names, four bounding box points, and the object score
    def ssd_mobilenet_inference(self, input_image):
        # Resize the image, convert to fp32, then transpose to CHW
        frame = input_image
        image_h = frame.shape[0]
        image_w = frame.shape[1]

        input_image = cv2.resize(input_image, (self.ssd_w, self.ssd_h), cv2.INTER_LINEAR)
        input_image = input_image.astype(np.float32)
        input_image = np.transpose(input_image, (2,0,1))

        # Performs the inference
        self._ssd_exec_net.start_async(request_id=ssd_mobilenet_processor.cur_request_id, inputs={self._ssd_input_blob: input_image})
        self._filtered_objs = []
        # wait for inference to complete
        if self._ssd_exec_net.requests[ssd_mobilenet_processor.cur_request_id].wait(-1) == 0:
            # get the inference result
            inference_results = self._ssd_exec_net.requests[ssd_mobilenet_processor.cur_request_id].outputs[self._ssd_output_blob]
            # process the results
            for num, detection_result in enumerate(inference_results[0][0]):
                # Draw only detection_resultects when probability more than specified threshold
                if detection_result[2] > ssd_mobilenet_processor.detection_threshold:
                    box_left = int(detection_result[3] * image_w)
                    box_top = int(detection_result[4] * image_h)
                    box_right = int(detection_result[5] * image_w)
                    box_bottom = int(detection_result[6] * image_h)
                    class_id = int(detection_result[1])
                    class_score = float(detection_result[2])
                    
                    self._filtered_objs.append(0)
                    self._filtered_objs.append(class_id)
                    self._filtered_objs.append(class_score)
                    self._filtered_objs.append(box_left)
                    self._filtered_objs.append(box_top)
                    self._filtered_objs.append(box_right)
                    self._filtered_objs.append(box_bottom)
                    
        
        return self._filtered_objs
        

