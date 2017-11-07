#include <iostream>
#include "Tensor.h"
int main() {
	int shape[] = {7, 7};
	Tensor ts(2, shape);
	ts.load_file("C:\\Users\\KnightJun\\OneDrive\\code\\cppdnn\\python\\test.antts") ;
	std::cout << ts;
	system("pause");
	return 0;
}