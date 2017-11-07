#pragma once
#include "Tensor.h"
namespace antdnn
{
	enum Conv_Flat
	{
		PADDING_SAME,
		PADDING_VALID
	};

	struct Conv2D_params
	{
		int kernel_count = -1;
		size2D kernel_size;
		int padding = PADDING_VALID;
	};
	DLL_ANTDNN_API void Conv2D(
		Tensor &in_tensor, Tensor &out_tensor, 
		Tensor &weights, Tensor &bias,
		int padding_type = PADDING_VALID);
	DLL_ANTDNN_API void Conv2DTranspose(
		Tensor &in_tensor, Tensor &out_tensor,
		Tensor &weights, Tensor &bias,
		int padding_type = PADDING_VALID,
		int strides_x=1, int strides_y=1);
	DLL_ANTDNN_API void Cropping2D(
		Tensor &in_tensor, Tensor &out_tensor,
		int top, int bottom, int left, int right
	);
}