#include "../headers/top_file.h"


NeuralNetwork::NeuralNetwork() {
    // The constructor initializes an empty layers vector
    layers.clear();
}

void NeuralNetwork::addLayer(int input_size, int output_size, std::function<double(double)> activation_func, std::function<double(double)> activation_func_derivative) {
    Layer newLayer(input_size, output_size, activation_func, activation_func_derivative);
    layers.push_back(newLayer);
}

void NeuralNetwork::forwardPropagation(const std::vector<double>& input) {
    // Ensure input size matches the input size of the first layer
    if (input.size() != layers[0].input_size) {
        std::cerr << "Input size does not match the input size of the first layer." << std::endl;
        return;
    }

    // Set the input to the first layer
    layers[0].forward(input);

    // Iterate through the layers and perform forward propagation
    for (size_t i = 1; i < layers.size(); ++i) {
        // Use the outputs of the previous layer as input for the current layer
        layers[i].forward(layers[i - 1].outputs);
    }
}


void NeuralNetwork::backPropagation(double y, const std::vector<double>& input) {
    // Calculate the output layer's error and delta
    Layer& outputLayer = layers.back();
    std::vector<double> outputLayerError(outputLayer.output_size, 0.0);

    for (int i = 0; i < outputLayer.output_size; ++i) {
        double output = outputLayer.outputs[i];
        double error = y - output;
        outputLayerError[i] = error * outputLayer.activation_func_derivative(output);

        // Update weights and biases for the output layer
        for (int j = 0; j < outputLayer.input_size; ++j) {
            outputLayer.weights[i][j] += LEARNING_RATE * outputLayerError[i] * layers[layers.size() - 2].outputs[j];
        }
    }

    // Propagate the error backward through the hidden layers
    for (int l = layers.size() - 2; l >= 0; --l) {
        Layer& hiddenLayer = layers[l];
        Layer& nextLayer = layers[l + 1];
        std::vector<double> hiddenLayerError(hiddenLayer.output_size, 0.0);

        // Calculate the hidden layer's error and delta
        for (int i = 0; i < hiddenLayer.output_size; ++i) {
            double output = hiddenLayer.outputs[i];
            double error = 0.0;
            for (int j = 0; j < nextLayer.output_size; ++j) {
                error += nextLayer.weights[j][i] * outputLayerError[j];
            }
            hiddenLayerError[i] = error * hiddenLayer.activation_func_derivative(output);

            // Update weights and biases for the hidden layer
            for (int k = 0; k < hiddenLayer.input_size; ++k) {
                hiddenLayer.weights[i][k] += LEARNING_RATE * hiddenLayerError[i] * (l > 0 ? layers[l - 1].outputs[k] : input[k]);
            }
        }
        outputLayerError = hiddenLayerError;
    }
}


void NeuralNetwork::train(const std::vector<std::vector<double>>& input_data, const std::vector<double>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;

        for (size_t i = 0; i < input_data.size(); ++i) {
            // Forward propagation
            forwardPropagation(input_data[i]);

            // Backpropagation
            backPropagation(y[i], input_data[i]);

            // Calculate Mean Squared Error (MSE) for monitoring training progress
            double error = 0.0;
            for (size_t j = 0; j < layers.back().outputs.size(); ++j) {
                error += 0.5 * std::pow(layers.back().outputs[j] - y[i], 2);
            }
            totalError += error;
        }

        // Calculate and print the average MSE for the epoch
        double averageError = totalError / input_data.size();
        std::cout << "Epoch " << epoch + 1 << ", \033[38;5;164mAverage MSE: \033[0m" << averageError << std::endl;
    }
}