#pragma once
#include "Tensor.h"
namespace antdnn
{
	DLL_ANTDNN_API void add(int ts_amount, Tensor **addvec, Tensor& ts_out);
}