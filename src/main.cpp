#include "../headers/top_file.h"

int raw_input_data[SIZE_DATASET][3];
int results[SIZE_DATASET];
double converted_input_data[SIZE_DATASET][3];

int main()
{
	std::string line = "";
	std::ifstream file("C:/Users/Nikita/Source/Repos/Small_DNN_3_inputs_2_hidden_layers/dataset.csv");

	//Checking whether the Dataset exists or not
	if (!file.is_open()) 
	{
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
		results[j] = (std::stoi(cell));
		j++;
		
	}
	// Close the file
	file.close();



	normalization_2d(raw_input_data, converted_input_data);


	return 0;
}
