#include "../headers/top_file.h"

// Sigmoid function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}