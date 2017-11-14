#include "TensorOperating.h"
#include "RunTimeDetect.h"

#include <Eigen/Eigen>
using namespace Eigen;
void antdnn::Matrix_Mul(Tensor &left, Tensor &right, Tensor &result)
{
	TimeDetect_BEG;
	int left_shape[2] = { left.size() / left.shape()[left.dim() - 1],
		left.shape()[left.dim() - 1] };
	int right_shape[2] = { right.size() / right.shape()[right.dim() - 1],
		right.shape()[right.dim() - 1] };
	assert(result.size() == left_shape[0] * right_shape[1]);
	Map<Matrix<float, -1, -1, RowMajor>> mat_left((float *)left.ptr_read(),
		left_shape[0], left_shape[1]);
	Map<Matrix<float, -1, -1, RowMajor>> mat_right((float *)right.ptr_read(),
		right_shape[0], right_shape[1]);

	Map<Matrix<float, -1, -1, RowMajor>> resmat((float *)result.ptr_write(),
		left_shape[0], right_shape[1]);
	resmat.noalias() = mat_left * mat_right;
	TimeDetect_END_Matrix_Mul;
	return;
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
void antdnn::Tensor2ConvMat(Tensor & input, Tensor & output,
	const int kernel_h, const int kernel_w, const int pad_h, const int pad_w)
{
	TimeDetect_BEG;
	const int stride_h = 1, dilation_h = 1;
	const int stride_w = 1, dilation_w = 1;
	auto inshape = input.shape();
	const int output_h = (inshape[0] + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (inshape[1] + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int dst_shape[2] = {output_h * output_w, kernel_h*kernel_w * inshape[2]};
	Tensor mid_ts(2, dst_shape);
	int in_cols = inshape[1] * inshape[2];
	int kernel_cols = inshape[2] * kernel_w;
	auto data_im = input.ptr_read();
	auto data_out = mid_ts.ptr_write();
	//for (auto i_out = 0; i_out < dst_shape[0]; i_out++)
	for(int row_in = -pad_h; row_in < output_h - pad_h; row_in++)
	for (int col_in = -pad_w; col_in < output_w - pad_w; col_in++)
	{
		//int row_in = i_out / output_w - pad_h, col_in = i_out % output_w - pad_w;

		for (int row_kernel = row_in; row_kernel < row_in + kernel_h; row_kernel++)
		{
			if (!is_a_ge_zero_and_a_lt_b(row_kernel, inshape[0]))
			{
				for (int i = kernel_cols; i--; *(data_out++) = 0);
			}
			else
			{
				int col_kernel = col_in * inshape[2];
				int col_kernel_end = col_kernel + kernel_cols;
				// count in input on convtion
				for (; col_kernel < col_kernel_end; col_kernel++)
				{
					if (!is_a_ge_zero_and_a_lt_b(col_kernel, in_cols))
					{
						*(data_out++) = 0;
					}
					else
					{
						*(data_out) = data_im[row_kernel * in_cols + col_kernel];
						data_out++;
					}
				}
			}
		}
	}
	output = mid_ts;
	TimeDetect_END_Tensor2ConvMat;
	return;
}

