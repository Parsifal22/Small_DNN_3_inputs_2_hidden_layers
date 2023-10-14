#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <stdlib.h>

#define SIZE_DATASET 10
#define NUM_OF_INPUTS 3
#define NUM_OF_HID1_NODES 5
#define NUM_OF_HID2_NODES 4
#define NUM_OF_OUTPUTS 1

void create_dataset();
void normalization_2d(int (*)[3], double (*)[3]);
int find_max(int(*)[3], int);
int find_min(int(*)[3], int);

void initialize_weights(int INPUT, int OUTPUT, double **w, double *b);

double sigmoid(double x);
double sigmoid_derivative(double x);

double relu(double x);
double relu_derivative(double x);