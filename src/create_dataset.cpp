#include "../headers/top_file.h"


void create_dataset() 
{
	std::ofstream myfile("C:/Users/Nikita/Source/Repos/Small_DNN_3_inputs_2_hidden_layers/dataset.csv");

    if (myfile.is_open())
    {   
        //Generator of random values
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < SIZE_DATASET; i++)
        {
            // Generate temperature with Gaussian distribution
            int temperature = generateRandomGaussian(gen, 15, 10, -70, 60);
            // Generate humidity with Gaussian distribution
            int humidity = generateRandomGaussian(gen, 35, 20, 0, 100);
            // Generate air quality with Gaussian distribution
            int air_quality = generateRandomGaussian(gen, 120, 50, 0, 500);

            int is_satisfay = (-30 < temperature < 45 && humidity < 40 && air_quality < 150) ? 1 : 0;

            myfile << temperature << "," << humidity << "," << air_quality << "," << is_satisfay << "\n";
        }

        myfile.close();
    }
    else std::cout << "Unable to open file";
}


// Function to generate random values with Gaussian distribution
int generateRandomGaussian(std::mt19937& gen, int mean, int stddev, int min, int max) {
    std::normal_distribution<double> distribution(mean, stddev);
    int value;
    do {
        value = static_cast<int>(distribution(gen));
    } while (value < min || value > max);
    return value;
}