#include "../headers/top_file.h"


// ReLU function
double relu(double x) {
    return (x > 0) ? x : 0;
}

// Derivative of ReLU function
double relu_derivative(double x) {
    return (x > 0) ? 1 : 0;
}