#include "../headers/top_file.h"

void create_dataset() 
{
	std::ofstream myfile("C:/Users/Nikita/Source/Repos/Small_DNN_3_inputs_2_hidden_layers/dataset.csv");

    if (myfile.is_open())
    {   
        //Generator of random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> distribution_tm(-70, 60);
        std::uniform_int_distribution<int> distribution_hm(0, 100);
        std::uniform_int_distribution<int> distribution_am(0, 300); // large range of data. I had to reduce it to 300

        for (int i = 0; i < SIZE_DATASET; i++)
        {
            int temperature = distribution_tm(gen);
            int humidity = distribution_hm(gen);
            int air_quality = distribution_am(gen);
            int is_satisfay = (-30 < temperature < 45 && humidity < 40 && air_quality < 150) ? 1 : 0;

            myfile << temperature << "," << humidity << "," << air_quality << "," << is_satisfay << "\n";
        }

        myfile.close();
    }
    else std::cout << "Unable to open file";
}