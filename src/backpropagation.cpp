#include "../headers/top_file.h"


void backpropagation(double &y, double* input, double& output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3, double* hidden1, double* hidden2)
{
    // Calculate the output layer error
    double output_error = y - output;

    // Calculate the error for the second hidden layer
    double hidden2Error[NUM_OF_HID2_NODES];
    for (int i = 0; i < NUM_OF_HID2_NODES; i++) {
        hidden2Error[i] = 0;
        for (int j = 0; j < NUM_OF_OUTPUTS; j++) {
            hidden2Error[i] += output_error * W3[j][i];
        }
        hidden2Error[i] *= sigmoid_derivative(hidden2[i]);  // Apply derivative of activation function
    }

    // Calculate the error for the first hidden layer
    double hidden1Error[NUM_OF_HID1_NODES];
    for (int i = 0; i < NUM_OF_HID1_NODES; i++) {
        hidden1Error[i] = 0;
        for (int j = 0; j < NUM_OF_HID2_NODES; j++) {
            hidden1Error[i] += hidden2Error[j] * W2[j][i];
        }
        hidden1Error[i] *= relu_derivative(hidden1[i]);  // Apply derivative of activation function
    }

    // Update weights and biases for the output layer
    for (int i = 0; i < NUM_OF_OUTPUTS; i++) {
        for (int j = 0; j < NUM_OF_HID2_NODES; j++) {
            W3[i][j] += LEARNING_RATE * output_error;
        }
        b3[i] += LEARNING_RATE * output_error;
    }

    // Update weights and biases for the second hidden layer
    for (int i = 0; i < NUM_OF_HID2_NODES; i++) {
        for (int j = 0; j < NUM_OF_HID1_NODES; j++) {
            W2[i][j] += LEARNING_RATE * hidden2Error[i];
        }
        b2[i] += LEARNING_RATE * hidden2Error[i];
    }

    // Update weights and biases for the first hidden layer
    for (int i = 0; i < NUM_OF_HID1_NODES; i++) {
        for (int j = 0; j < NUM_OF_INPUTS; j++) {
            W1[i][j] += LEARNING_RATE * hidden1Error[i];
        }
        b1[i] += LEARNING_RATE * hidden1Error[i];
    }


}