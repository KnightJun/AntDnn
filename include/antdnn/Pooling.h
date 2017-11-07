#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void MaxPooling2D(Tensor & in_ts, Tensor & out_ts, int poolw, int poolh);
}