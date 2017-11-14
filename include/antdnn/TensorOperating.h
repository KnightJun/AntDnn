#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void Tensor2ConvMat(Tensor & input, Tensor & output,
		const int kernel_h, const int kernel_w, const int pad_h, const int pad_w);
        DLL_ANTDNN_API void Matrix_Mul(Tensor &left, Tensor &right, Tensor &result);
}
