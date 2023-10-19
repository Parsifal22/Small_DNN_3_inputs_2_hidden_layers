#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <filesystem>

#define SIZE_DATASET 500
#define EPOCHS 500
#define LEARNING_RATE 0.0001

#define NUM_OF_INPUTS 3
#define NUM_OF_HID1_NODES 5
#define NUM_OF_HID2_NODES 4
#define NUM_OF_OUTPUTS 1
 

void create_dataset();
int generateRandomGaussian(std::mt19937& gen, int mean, int stddev, int min, int max);

void normalization_2d(int **, double **);
int find_max(int **, int);
int find_min(int **, int);

void initialize_weights(int INPUT, int OUTPUT, double **w, double *b);

double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);

void forward_propagation(double *input, double &output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2);
void backpropagation(double &y, double *inputs,  double &output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2);

