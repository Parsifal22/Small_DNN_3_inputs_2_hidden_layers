#include "../headers/top_file.h"

void normalization_2d(std::vector<std::vector<int>>& input_data, std::vector<std::vector<double>>& output_data) {
    for (size_t i = 0; i < input_data[0].size(); ++i) {
        int max = find_max(input_data, i);
        int min = find_min(input_data, i);

        for (size_t j = 0; j < input_data.size(); ++j) {
            output_data[j][i] = static_cast<double>(input_data[j][i] - min) / (max - min);
        }
    }
}

int find_max(std::vector<std::vector<int>> input_data, size_t j) {
    int max = -9999999;
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (max < input_data[i][j]) {
            max = input_data[i][j];
        }
    }
    return max;
}

int find_min(std::vector<std::vector<int>> input_data, size_t j) {
    int min = 9999999;
    for (size_t i = 0; i < input_data.size(); ++i) {
        if (min > input_data[i][j]) {
            min = input_data[i][j];
        }
    }
    return min;
}