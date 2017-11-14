#pragma once
#include <time.h>
#include "antdnn_macro.h"
#if OpenRunTimeDetect == 1
#define TimeDetect_BEG clock_t __tr_start = clock();

extern double &tr_Tensor2ConvMat;
#define TimeDetect_END_Tensor2ConvMat tr_Tensor2ConvMat += (double)(clock() - __tr_start) / CLOCKS_PER_SEC;
extern double &tr_Matrix_Mul;
#define TimeDetect_END_Matrix_Mul tr_Matrix_Mul += (double)(clock() - __tr_start) / CLOCKS_PER_SEC;

DLL_ANTDNN_API void PrintRunTime();
#endif