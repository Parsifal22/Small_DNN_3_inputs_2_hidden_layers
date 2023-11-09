#include "../headers/top_file.h"


Layer::Layer(int input_size, int output_size, std::function<double(double)> activation_func, std::function<double(double)> activation_func_derivative) :
input_size(input_size), output_size(output_size), activation_function(activation_func), activation_func_derivative(activation_func_derivative) {
    weights.resize(output_size, std::vector<double>(input_size, 0.0));
    outputs.resize(output_size, 0.0);
    initializeWeightsAndBiases();
}


void Layer::initializeWeightsAndBiases() {

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Calculate the He initialization factor
    double scale = std::sqrt(2.0 / input_size);

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Generate a random weight value with mean 0 and standard deviation 'scale'
            weights[i][j] = scale * (static_cast<double>(std::rand()) / RAND_MAX);
        }
        outputs.push_back(0.0);
    }
}

void Layer::forward(const std::vector<double>& input) {
    // Ensure input size matches the expected input size for this layer
    if (input.size() != input_size) {
        std::cerr << "Input size does not match layer's input size." << std::endl;
        return;
    }

    // Clear previous outputs
    outputs.clear();

    // Calculate the weighted sum of inputs and apply the sigmoid activation function
    for (int i = 0; i < output_size; ++i) {
        double weighted_sum = 0.0;
        for (int j = 0; j < input_size; ++j) {
            weighted_sum += input[j] * weights[i][j];
        }
        double output = activation_function(weighted_sum);
        outputs.push_back(output);
    }
}
