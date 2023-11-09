#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <filesystem>
#include <functional>


#define SIZE_DATASET 10
#define EPOCHS 50
#define LEARNING_RATE 0.001

#define NUM_OF_INPUTS 3
#define NUM_OF_HID1_NODES 5
#define NUM_OF_HID2_NODES 4
#define NUM_OF_OUTPUTS 1


// Define prototype for class Layer and NeuralNetwork
class Layer {
public:
    int input_size;
    int output_size;
    std::vector<std::vector<double>> weights;
    std::vector<double> outputs;
    std::function<double(double)> activation_function;
    std::function<double(double)> activation_func_derivative;

    Layer(int input_size, int output_size, std::function<double(double)> activation_func, std::function<double(double)> activation_func_derivative);
    void forward(const std::vector<double>& input);
    void initializeWeightsAndBiases();


};

class NeuralNetwork {
public:

    std::vector<Layer> layers;

    NeuralNetwork();
    void forwardPropagation(const std::vector<double>& input);
    void backPropagation(double y, const std::vector<double>& input);
    void addLayer(int input_size, int output_size, std::function<double(double)> activation_func, std::function<double(double)> activation_func_derivative);
    void train(const std::vector<std::vector<double>>& input_data, const std::vector<double>& y, int epochs);

};


void create_dataset();

int generateRandomGaussian(std::mt19937& gen, int mean, int stddev, int min, int max);

int find_max(std::vector<std::vector<int>> input_data, size_t j);
int find_min(std::vector<std::vector<int>> input_data, size_t j);
void normalization_2d(std::vector<std::vector<int>>& input_data, std::vector<std::vector<double>>& output_data);


void forward_propagation(double *input, double &output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2);
void backpropagation(double &y, double *inputs,  double &output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2);

