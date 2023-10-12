#include "../headers/top_file.h"


int main()
{
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
	

	
	return 0;
}
