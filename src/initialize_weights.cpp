/*
He Initialization: This method is suitable for layers where the activation function is a ReLU function. 
The weights are initialized from a normal distribution with mean 0 and standard deviation sqrt(2 / (fan_in)),
where fan_in is the number of input nodes https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/.
*/

#include "../headers/top_file.h"

// He Initialization
void initialize_weights(int INPUT, int OUTPUT, double **w, double *b) {

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Calculate the He initialization factor
    double scale = std::sqrt(2.0 / INPUT);

    for (int i = 0; i < OUTPUT; i++) {
        for (int j = 0; j < INPUT; j++) {
            // Generate a random weight value with mean 0 and standard deviation 'scale'
            w[i][j] = scale * (static_cast<double>(std::rand()) / RAND_MAX);
        }
        b[i] = 0;
    }

}