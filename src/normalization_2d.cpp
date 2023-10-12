#include "../headers/top_file.h"

void normalization_2d(int (*input_data)[3], double (*output_data)[3]) 
{
    double sum;
    for (int i = 0; i < 3; i++) {
        sum = 0;
        for (int j = 0; j < SIZE_DATASET; j++) {
            sum += abs(input_data[j][i]);
        }
        for (int j = 0; j < SIZE_DATASET; j++) {
            output_data[j][i] = abs(input_data[j][i]) / sum;
        }
    }
}