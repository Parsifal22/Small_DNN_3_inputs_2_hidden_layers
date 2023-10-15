#include "../headers/top_file.h"


void forward_propagation(double** input, double* output, double** W1, double** W2, double** W3, double* b1, double* b2, double* b3) {

    for(int k = 0; k < SIZE_DATASET; k++)
    {
		double input_line[NUM_OF_INPUTS];
		for(int i = 0; i < NUM_OF_INPUTS; i++)
		{
			input_line[i] = input[k][i];
		}

	    // Hidden layer 1
	    double hidden1[NUM_OF_HID1_NODES];
	    for (int i = 0; i < NUM_OF_HID1_NODES; i++) {
	        hidden1[i] = 0;
	        for (int j = 0; j < NUM_OF_INPUTS; j++) {
	            hidden1[i] += W1[i][j] * input_line[j];
	        }
	        hidden1[i] += b1[i];
	        hidden1[i] = relu(hidden1[i]);
	    }

		std::cout << "\033[38;5;68mFirst Hidden Layer:\033[0m" << std::endl;
		for (int i = 0; i < NUM_OF_HID1_NODES; i++) {
	
			std::cout << hidden1[i] << "\t";

		}

	    // Hidden layer 2
	    double hidden2[NUM_OF_HID2_NODES];

	    for (int i = 0; i < NUM_OF_HID2_NODES; i++) {
	        hidden2[i] = 0;
	        for (int j = 0; j < NUM_OF_HID1_NODES; j++) {
	            hidden2[i] += W2[i][j] * hidden1[j];
	        }
	        hidden2[i] += b2[i];
	        hidden2[i] = relu(hidden2[i]);
	    }

		std::cout << "\n\033[38;5;100mSecond Hidden Layer:\033[0m" << std::endl;
		for (int i = 0; i < NUM_OF_HID2_NODES; i++) {

			std::cout << hidden2[i] << "\t";

		}


	    // Output layer
	    for (int i = 0; i < NUM_OF_OUTPUTS; i++) {
	        output[k] = 0;
	        for (int j = 0; j < NUM_OF_HID2_NODES; j++) {
	            output[k] += W3[i][j] * hidden2[j];
	        }
	        output[k] += b3[i];
	        output[k] = sigmoid(output[k]);
	    }

		std::cout << "\n\033[38;5;103mOutput:\033[0m" << std::endl;

    	std::cout << output[k] << std::endl;

    }


}