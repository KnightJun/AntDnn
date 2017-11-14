#include <iostream>
#include "RunTimeDetect.h"
double tr_record[2] = { 0 };
double &tr_Tensor2ConvMat = tr_record[0];
double &tr_Matrix_Mul = tr_record[1];
using namespace std;

DLL_ANTDNN_API void PrintRunTime()
{

	cout << "============Time record============" << endl;
	cout << "Tensor2ConvMat: " << tr_Tensor2ConvMat << endl;
	cout << "Matrix_Mul: " << tr_Matrix_Mul << endl;
	cout << "===================================" << endl;
	return void();
}
