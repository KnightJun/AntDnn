#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void Dense(Tensor & in_ts, Tensor & out_ts, Tensor &weights, Tensor &bias);
}