#include "../headers/top_file.h"


int main()
{

    // Define lambda functions for activation functions
    auto sigmoid = [](double x) -> double {
        return 1.0 / (1.0 + std::exp(-x));
    };

    auto relu = [](double x) -> double {
        return std::max(0.0, x);
    };

    auto sigmoid_derivative = [](double x) -> double {
        double sigmoid_x = 1.0 / (1.0 + std::exp(-x));
        return sigmoid_x * (1.0 - sigmoid_x);
    };

    auto relu_derivative = [](double x) -> double {
        return (x > 0.0) ? 1.0 : 0.0;
    };

    std::vector<std::vector<int>>  raw_input_data;
    std::vector<std::vector<double>> converted_input_data;
    converted_input_data.resize(SIZE_DATASET, std::vector<double>(3, 0.0));
    std::vector<double> y;

    std::filesystem::path full_path;

    try {
        std::filesystem::path current_path = std::filesystem::current_path();

        // Check if current_path is not null before using it
        if (!current_path.empty()) {
            full_path = current_path.parent_path().parent_path().parent_path() / "dataset.csv";

        }
        else {
            std::cerr << "Error: current_path is null." << std::endl;
        }

    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

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

            // Iterate through each line and split the content using delimiter
            std::string line = "";

            while (std::getline(file, line)) {
                std::stringstream lineStream(line);
                std::string cell;
                int i = 0;

                std::vector<int> input_row;

                // Iterate through each cell and convert the value to integer
                for (int i = 0; i < 3; i++) {
                    std::getline(lineStream, cell, ',');
                    input_row.emplace_back(std::stoi(cell));
                }

                raw_input_data.emplace_back(std::move(input_row));

                std::getline(lineStream, cell, ',');
                y.emplace_back(std::stod(cell));
            }

            // Close the file
            file.close();

        }
    }

    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    normalization_2d(raw_input_data, converted_input_data);

    auto dnn = NeuralNetwork();
    dnn.addLayer(NUM_OF_INPUTS, NUM_OF_HID1_NODES, relu, relu_derivative);
    dnn.addLayer(NUM_OF_HID1_NODES, NUM_OF_HID2_NODES, relu, relu_derivative);
    dnn.addLayer(NUM_OF_HID2_NODES, NUM_OF_OUTPUTS, sigmoid, sigmoid_derivative);

    dnn.train(converted_input_data, y, EPOCHS);


    return 0;
}