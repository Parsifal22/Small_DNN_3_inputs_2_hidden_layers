#include "../headers/top_file.h"

void normalization_2d(int(*input_data)[3], double(*output_data)[3])
{
    int max;
    int min;
    for (int i = 0; i < 3; i++) {
        max = find_max(input_data, i);
        min = find_min(input_data, i);

        for (int j = 0; j < SIZE_DATASET; j++) {

            output_data[j][i] = (double) abs(input_data[j][i]) / (max - min);
        }
    }
}

int find_max(int (*input_data)[3], int j)
{
    int max = -9999999;
    for(int i=0; i < SIZE_DATASET; i++)
    {
	    if(max < input_data[i][j])
	    {
            max = input_data[i][j];
	    }
    }
    return max;
}



int find_min(int(*input_data)[3], int j)
{
    int min = 9999999;
    for (int i = 0; i < SIZE_DATASET; i++)
    {
        if (min > input_data[i][j])
        {
            min = input_data[i][j];
        }
    }
    return min;
}