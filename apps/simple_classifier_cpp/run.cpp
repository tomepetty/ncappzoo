// Simple classifier cpp

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <vector>

using namespace InferenceEngine;

#define DEVICE "MYRIAD"

std::vector<std::string> labels;
const unsigned int MAX_PATH = 256;

// *************************************************************************
// Read the network labels from the provided labels file
// *************************************************************************
void getNetworkLabels(std::string labelsDir, std::vector<std::string>* labelsVector)
{
    char filename[MAX_PATH];
    strncpy(filename, labelsDir.c_str(), MAX_PATH);
    FILE* cat_file = fopen(filename, "r");
    if (cat_file == nullptr) {
        std::cerr << "Could not find Category file." << std::endl;
        exit(1);
    }

    char cat_line[255];
    while (fgets(cat_line , 255 , cat_file) != NULL) {
        if (cat_line[strlen(cat_line) - 1] == '\n')
            cat_line[strlen(cat_line) - 1] = '\0';
        labelsVector->push_back(std::string(cat_line));
    }
    fclose (cat_file);
}

// *************************************************************************
// Entrypoint for the application
// *************************************************************************
int main(int argc, char *argv[]) {
    cv::Mat imgIn;
    cv::Mat imgInput;
    
    if (argc != 4) {
        std::cout << " ./simple_classifier_cpp <XML FILE> <IMAGE> <LABELS> ";
        exit(1);
    }
    // Get all of the parameters that we need to run the inference
    std::string XML = argv[1];
    std::string BIN = XML.substr(0, XML.length()-3) + "bin";
    std::string IMAGE = argv[2];
    std::string LABELS = argv[3];
    
    // Read the labels
    getNetworkLabels(LABELS, &labels);
    
    // ----------------------- Create IE core object and read the network ----------------------- //
    // Create the inference engine core object
    Core ieCore;
    // Create a network reader and read in the network and weights
    CNNNetReader networkReader;
    networkReader.ReadNetwork(XML);
    networkReader.ReadWeights(BIN);
    auto network = networkReader.getNetwork();
    
    // ----------------------- Set up the network input ----------------------- //
    InputsDataMap inputDataMap(network.getInputsInfo());
    InputInfo::Ptr& inputInfo = inputDataMap.begin()->second;
    // Get the input node name
    std::string inputLayerName = inputDataMap.begin()->first;
    // Set precision for the input
    inputInfo->setPrecision(Precision::U8);

    // ----------------------- Set up the network output ----------------------- //
    OutputsDataMap outputDataMap(network.getOutputsInfo());
    auto outputData = outputDataMap.begin()->second;
    // Get the output node name
    std::string outputLayerName = outputDataMap.begin()->first;
    // Set precision for output
    outputData->setPrecision(Precision::FP32);
    
    // ----------------------- Load the network and create the inference request ----------------------- //
    // Load the network to the device (default: Myriad)
    auto executableNetwork = ieCore.LoadNetwork(network, DEVICE);
    // Create the inference request
    auto inferenceRequest = executableNetwork.CreateInferRequestPtr();
    
    // ----------------------- Prepare the input data ----------------------- //
    // Create buffer to hold input data
    auto inputBlob = inferenceRequest->GetBlob(inputLayerName);
    auto inputData = inputBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
    
    // Get the input dimensions for the network
    auto inputDims = inferenceRequest->GetBlob(inputLayerName)->getTensorDesc().getDims();
    unsigned int inputNumberOfChannels = inputDims.at(1);
    unsigned int inputHeight = inputDims.at(2);
    unsigned int inputWidth = inputDims.at(3);
    
    // Use OpenCV to read in an image
    imgIn = cv::imread(IMAGE);

    // Resize the input image in accordance to the network input size
    cv::resize(imgIn, imgInput, cv::Size(inputHeight, inputWidth));
    
    // Prepare to fill the buffer with the image data
    size_t imageSize = inputHeight * inputWidth;
    // Fills buffer with the image data. This data will be sent to the device for inference
    for (size_t pixelIndex = 0; pixelIndex < imageSize; ++pixelIndex) {
        for (size_t channel = 0; channel < inputNumberOfChannels; ++channel) {
            inputData[channel * imageSize + pixelIndex] = imgInput.at<cv::Vec3b>(pixelIndex)[channel];
        }
    }
    
    // ----------------------- Make the inference ----------------------- //
    inferenceRequest->Infer();
    
    // ----------------------- Process the results ----------------------- //
    // Get the inference results
    auto inferenceResults = inferenceRequest->GetBlob(outputLayerName);
    // Get all of the confidence scores. 
    auto scores = inferenceResults->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>(); 

    // Sort the results and get the number of desired top results
    std::vector<unsigned> sortedResults; // This vector will hold all of the top sorted results
    unsigned int resultsToDisplay = 5;   // How many results should return?
    TopResults(resultsToDisplay, *inferenceResults, sortedResults);

    // ----------------------- Display the results ----------------------- //  
    std::cout << std::endl << "\033[1;33m **********  Results  ***********\033[0m"<< std::endl << std::endl;
    for (size_t resultIndex = 0; resultIndex < resultsToDisplay; ++resultIndex) {
        auto confidenceScore = scores[sortedResults[resultIndex]] * 100;
        auto labelIndex = sortedResults.at(resultIndex);
        std::cout << " Prediction is " << std::setprecision(1) << std::fixed << confidenceScore << "% "  << labels[labelIndex] <<std::endl;
    }
    std::cout << "\n Finished." << '\n';
    
}
