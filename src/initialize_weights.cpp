/*
He Initialization: This method is suitable for layers where the activation function is a ReLU function. 
The weights are initialized from a normal distribution with mean 0 and standard deviation sqrt(2 / (fan_in)),
where fan_in is the number of input nodes https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/.
*/

#include "../headers/top_file.h"

// He Initialization
void initialize_weights(int INPUT, int OUTPUT, double **w, double *b) {

    double limit = sqrt(2.0 / INPUT);

    for (uint32_t i = 0; i < OUTPUT; i++) {
        for (uint32_t j = 0; j < INPUT; j++) {
            w[i][j] = (double)rand() / RAND_MAX * 2 * limit - limit;
        }
        b[i] = 0;
    }

}