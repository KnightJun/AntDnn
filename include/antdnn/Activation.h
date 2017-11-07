#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void activation_relu(Tensor & ts);
	DLL_ANTDNN_API void activation_softmax(Tensor & ts);
	DLL_ANTDNN_API void flatten(Tensor & ts);
}