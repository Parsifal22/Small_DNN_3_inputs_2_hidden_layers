#include "../headers/top_file.h"


int main()
{


	//Allocate memory for weights and biases
	int**  raw_input_data = new int* [SIZE_DATASET]; for (uint32_t i = 0; i < SIZE_DATASET; i++) { raw_input_data[i] = new int[3]; }
	int* y = new int[SIZE_DATASET];
	double *results = new double[SIZE_DATASET];

	double** converted_input_data = new double* [SIZE_DATASET]; for (uint32_t i = 0; i < SIZE_DATASET; i++) { converted_input_data[i] = new double[3]; }

	//Allocate memory for weights and biases
	double** W1 = new double* [NUM_OF_HID1_NODES]; for (uint32_t i = 0; i < NUM_OF_HID1_NODES; i++) { W1[i] = new double[NUM_OF_INPUTS]; }
	double** W2 = new double* [NUM_OF_HID2_NODES]; for (uint32_t i = 0; i < NUM_OF_HID2_NODES; i++) { W2[i] = new double[NUM_OF_HID1_NODES]; }
	double** W3 = new double* [NUM_OF_OUTPUTS]; for (uint32_t i = 0; i < NUM_OF_OUTPUTS; i++) { W3[i] = new double[NUM_OF_HID2_NODES]; }

	double *b1 = new double[NUM_OF_HID1_NODES];
	double *b2 = new double[NUM_OF_HID2_NODES];
	double *b3 = new double[NUM_OF_OUTPUTS];


	

	//Checking whether the Dataset exists or not
	std::ifstream file("C:/Users/Nikita/Source/Repos/Small_DNN_3_inputs_2_hidden_layers/dataset.csv");

	if (!file.is_open()) 
	{
		// Close the file
		file.close();
		create_dataset();
		std::ifstream file("C:/Users/Nikita/Source/Repos/Small_DNN_3_inputs_2_hidden_layers/dataset.csv");
		std::string const is_open = file.is_open() ? "File is open successfully" : "Error: file doesn't exist";
		std::cout << is_open << std::endl;
	}
	else
	{
		std::cout << "File is open successfully!" << std::endl;
	}

	int j = 0;

	// Iterate through each line and split the content using delimiter
	std::string line = "";

	while (std::getline(file, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		int i = 0;

		// Iterate through each cell and convert the value to integer
		for(int i = 0; i < 3; i++)
		{
			std::getline(lineStream, cell, ',');
			raw_input_data[j][i] = (std::stoi(cell));

		}

		std::getline(lineStream, cell, ',');
		y[j] = (std::stoi(cell));
		j++;
		
	}
	// Close the file
	file.close();


	initialize_weights(NUM_OF_INPUTS, NUM_OF_HID1_NODES, W1, b1);

	std::cout << "\033[34mW1 initialized\033[0m" << std::endl;

	for (uint32_t i = 0; i < NUM_OF_HID1_NODES; i++)
	{
		for(uint32_t j=0; j < NUM_OF_INPUTS; j++)
		{
			std::cout << W1[i][j] << "\t";
		}
		std::cout << b1[i] << std::endl;
	}

	initialize_weights(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, W2, b2);

	std::cout << "\033[34mW2 initialized\033[0m" << std::endl;

	for (uint32_t i = 0; i < NUM_OF_HID2_NODES; i++)
	{
		for (uint32_t j = 0; j < NUM_OF_HID1_NODES; j++)
		{
			std::cout << W2[i][j] << "\t";
		}
		std::cout << b2[i] << std::endl;
	}

	initialize_weights(NUM_OF_HID2_NODES, NUM_OF_OUTPUTS, W3, b3);

	std::cout << "\033[34mW3 initialized\033[0m" << std::endl;

	for (uint32_t i = 0; i < NUM_OF_OUTPUTS; i++)
	{
		for (uint32_t j = 0; j < NUM_OF_HID2_NODES; j++)
		{
			std::cout << W2[i][j] << "\t";
		}
		std::cout << b2[i] << std::endl;
	}



	normalization_2d(raw_input_data, converted_input_data);

	std::cout << "\033[35mNormalized data:\033[0m" << std::endl;
	for(int i=0; i < SIZE_DATASET; i++)
	{
		for(int j=0; j < 3; j++)
		{
			std::cout << converted_input_data[i][j] << "\t";
		}
		std::cout << y[i] << std::endl;
	}

	forward_propagation(converted_input_data, results, W1, W2, W3, b1, b2, b3);

	std::cout << "\n\033[38;5;45mAll results:\033[0m" << std::endl;
	for (int i = 0; i < SIZE_DATASET; i++)
	{
		std::cout << results[i] << "\t";
	}





	for (uint32_t i = 0; i < SIZE_DATASET; i++) { delete[] raw_input_data[i]; }
	delete[] raw_input_data;

	for (uint32_t i = 0; i < SIZE_DATASET; i++) { delete[] converted_input_data[i]; }
	delete[] converted_input_data;

	delete[] y;
	delete[] results;

	// Deallocate memory for weights and biases

	for (uint32_t i = 0; i < NUM_OF_HID1_NODES; i++) { delete[] W1[i]; }
	delete[] W1;

	for (uint32_t i = 0; i < NUM_OF_HID2_NODES; i++) { delete[] W2[i]; }
	delete[] W2;

	for (uint32_t i = 0; i < NUM_OF_OUTPUTS; i++) { delete[] W3[i]; }
	delete[] W3;

	delete[] b1;
	delete[] b2;
	delete[] b3;

	return 0;
}
