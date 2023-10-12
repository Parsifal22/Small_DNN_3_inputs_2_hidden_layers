#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <stdlib.h>

#define SIZE_DATASET 10

void create_dataset();
void normalization_2d(int (*)[3], double (*)[3]);