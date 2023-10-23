#include "../headers/top_file.h"


int main()
{


	//Allocate memory for input data
	int**  raw_input_data = new int* [SIZE_DATASET]; for (uint32_t i = 0; i < SIZE_DATASET; i++) { raw_input_data[i] = new int[3]; }
	double* y = new double[SIZE_DATASET];
	double *results = new double[SIZE_DATASET];

	double** converted_input_data = new double* [SIZE_DATASET]; for (uint32_t i = 0; i < SIZE_DATASET; i++) { converted_input_data[i] = new double[3]; }

	//Allocate memory for weights and biases
	double** W1 = new double* [NUM_OF_HID1_NODES]; for (uint32_t i = 0; i < NUM_OF_HID1_NODES; i++) { W1[i] = new double[NUM_OF_INPUTS]; }
	double** W2 = new double* [NUM_OF_HID2_NODES]; for (uint32_t i = 0; i < NUM_OF_HID2_NODES; i++) { W2[i] = new double[NUM_OF_HID1_NODES]; }
	double** W3 = new double* [NUM_OF_OUTPUTS]; for (uint32_t i = 0; i < NUM_OF_OUTPUTS; i++) { W3[i] = new double[NUM_OF_HID2_NODES]; }

	double *b1 = new double[NUM_OF_HID1_NODES];
	double *b2 = new double[NUM_OF_HID2_NODES];
	double *b3 = new double[NUM_OF_OUTPUTS];



	std::filesystem::path current_path = std::filesystem::current_path();
	std::filesystem::path full_path = current_path.parent_path().parent_path().parent_path() / "dataset.csv";

	if (std::filesystem::exists(full_path)) 
	{
		std::cout << "Found file at: " << full_path << std::endl;
	}
	else
	{
		std::cerr << "File not found at: " << full_path << std::endl;
		std::cerr << "Creating file dataset.csv... " << full_path << std::endl;
		create_dataset();
	}


	//Read data from file dataset.csv
	try {
		std::ifstream file(full_path);

		if (!file.is_open()) {
			throw std::runtime_error("Error: File doesn't exist at " + full_path.string());
		}
		else {
			std::cout << "File is open successfully!" << std::endl;

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
				y[j] = (std::stod(cell));
				j++;
				
			}
			// Close the file
			file.close();
				}
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
	}



	initialize_weights(NUM_OF_INPUTS, NUM_OF_HID1_NODES, W1, b1);
	for(int i=0; i < NUM_OF_HID1_NODES; i++)
	{
		for(int j=0; j < NUM_OF_INPUTS; j++)
		{
			std::cout << W1[i][j] << "\t";
		}
		std::cout << b1[i] << "\n";

	} 
	initialize_weights(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, W2, b2);
	initialize_weights(NUM_OF_HID2_NODES, NUM_OF_OUTPUTS, W3, b3);

	normalization_2d(raw_input_data, converted_input_data);


	for (uint32_t i = 0; i < EPOCHS; i++)
	{
		std::cout << "\n\n\n\033[38;5;196mThis is the epoch: " << i << "\033" <<  std::endl;
		for(uint32_t j = 0; j < SIZE_DATASET; j++)
		{
			double* hidden1 = new double [NUM_OF_HID1_NODES];
			double* hidden2 = new double[NUM_OF_HID2_NODES];

			double* input_line = new double[NUM_OF_INPUTS];

			for (int k = 0; k < NUM_OF_INPUTS; k++)
			{
				input_line[k] = converted_input_data[j][k];
			}
			double &result = results[j];
			forward_propagation(input_line, result, W1, W2, W3, b1, b2, b3, hidden1, hidden2);


			double& y_1 = y[j];
			backpropagation(y_1, input_line, result, W1, W2, W3, b1, b2, b3, hidden1, hidden2);

			delete[] hidden1;
			delete[] hidden2;
			delete[] input_line;
		}

		//Mean Squared Error(MSE) Cost Function
		std::cout << "\n\033[38;5;164mCost function:\033[0m" << "\t";
		double sum = 0;
		for(int i =0; i < SIZE_DATASET; i++)
		{
			sum += pow(y[i] - results[i], 2);
		}
		
		std::cout << sum/SIZE_DATASET << std::endl;
	}



	// Deallocate memory for input and output data

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
