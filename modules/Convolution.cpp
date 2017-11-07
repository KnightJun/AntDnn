#include "Convolution.h"
#define forrange(i, d) for(int i = 0; i < d; i++)
#include <assert.h>
void antdnn::Conv2D(
	Tensor & in_tensor, Tensor & out_ts, 
	Tensor &weights, Tensor &bias,
	int padding_type)
{
	Conv2D_params params;
	auto wei_shape = weights.shape();
	params.kernel_size.h = wei_shape[0];
	params.kernel_size.w = wei_shape[1];
	params.kernel_count = wei_shape[3];
	params.padding = padding_type;
	Tensor mid_tensor = in_tensor, out_tensor;
	int top = (params.kernel_size.h - 1) / 2,
		bottom = top,
		left = (params.kernel_size.w - 1) / 2,
		right = left;
	if (params.padding == PADDING_SAME)
	{
		mid_tensor.makeBorder2D(top, bottom, left, right, 0);
	}
	auto src_shape = mid_tensor.shape();
	int dst_shape[3] = { 
		src_shape[0] - (params.kernel_size.h - 1), 
		src_shape[1] - (params.kernel_size.w - 1) , 
		params.kernel_count };
	out_tensor.recreate(3, dst_shape);
	out_tensor.set_to(0);
	// 3, 3, 3, 32
	float *calc_res = out_tensor.ptr_write();

	// get the row ptr vector from srouce matrix
	int old_rows = src_shape[1] * src_shape[2];
	int kernel_rows = params.kernel_size.w * src_shape[2] * params.kernel_count;
	const float **rptrs_src = new const float*[params.kernel_size.h];
	const float **rptrs_kernel = new const float*[params.kernel_size.h];
	auto rptrs_src_release = rptrs_src, rptrs_kernel_release = rptrs_kernel;
	const float *ptr_bias = bias.ptr_read();
	for (int i = 0; i < dst_shape[0]; i ++)
	{
		// set src rows ptr
		rptrs_src[0] = mid_tensor.ptr_read() + i * old_rows;
		for (int i = 1; i < params.kernel_size.h; i++)
		{
			rptrs_src[i] = rptrs_src[i - 1] + old_rows;
		}

		for (int j = 0; j < dst_shape[1]; j++)
		{
			// reset kernel row point
			rptrs_kernel[0] = weights.ptr_read();
			for (int i_r = 1; i_r < params.kernel_size.h; i_r++)
			{
				rptrs_kernel[i_r] = rptrs_kernel[i_r - 1] + kernel_rows;
			}

			for (int j_kernel = 0; j_kernel < params.kernel_size.w; j_kernel++) // 3
			{
				for (int c_src = 0; c_src < src_shape[2]; c_src++) // 3
				{
					for (int c_dst = 0; c_dst < params.kernel_count; c_dst++) // 32
					{
						for (int i_kernel = 0; i_kernel < params.kernel_size.h; i_kernel++) // 3
						{
							calc_res[c_dst] += rptrs_src[i_kernel][c_src] * (*(rptrs_kernel[i_kernel]++));
						}
					}

				}
				for (int i_r = 0; i_r < params.kernel_size.h; i_r++)
				{
					rptrs_src[i_r] += src_shape[2];
				}
			}
			//add bias
			for (int c_dst = 0; c_dst < params.kernel_count; c_dst++) // 32
			{
				// output matrix next data
				*(calc_res++) += ptr_bias[c_dst];
			}
			// turn back the rows point 
			for (int i_r = 0; i_r < params.kernel_size.h; i_r++)
			{
				rptrs_src[i_r] -= src_shape[2] * (params.kernel_size.w - 1);
			}
		}
	}
	delete[] rptrs_src_release;
	delete[] rptrs_kernel_release;
	out_ts = out_tensor;
}

void antdnn::Conv2DTranspose(Tensor & in_tensor, Tensor & out_ts,
	Tensor & weights, Tensor & bias,
	int padding_type, int strides_x, int strides_y)
{
	auto src_shape = in_tensor.shape();
	auto kernel_shape = weights.shape();
	int &kernel_h = kernel_shape[0];
	int &kernel_w = kernel_shape[1];
	int &kernel_c = kernel_shape[2];
	assert(kernel_shape[3] == src_shape[2]);
	int dst_shape[3] = {
		(src_shape[0] - 1) * strides_y + 1 + (kernel_h - 1),
		(src_shape[1] - 1) * strides_x + 1 + (kernel_w - 1),
		kernel_c };
	Tensor mid_tensor = in_tensor;
	Tensor out_tensor(3, dst_shape);
	out_tensor.set_to(0);
	assert(padding_type == PADDING_VALID);
	// init ptr
	int rows_kernel = kernel_w*kernel_c * src_shape[2];
	int rows_dst = dst_shape[1] * dst_shape[2];
	int dst_step_x = - (kernel_c * (kernel_w - strides_x));
	int dst_step_y = (kernel_c * (kernel_w - strides_x)) + rows_dst * (strides_y - 1);

	const float **kernel_ptrs = new const float*[kernel_h];
	auto kernel_ptrs_release = kernel_ptrs; // use for delete
	const float **kernel_ptrs_bak = new const float*[kernel_h];
	float **dst_ptrs = new float*[kernel_h];
	auto dst_ptrs_release = dst_ptrs; // use for delete
	kernel_ptrs[0] = weights.ptr_read();
	dst_ptrs[0] = out_tensor.ptr_write();
	for (int i = 1; i < kernel_h; i++)
	{
		kernel_ptrs[i] = kernel_ptrs[i - 1] + rows_kernel;
		dst_ptrs[i] = dst_ptrs[i - 1] + rows_dst;
	}
	memcpy(kernel_ptrs_bak, kernel_ptrs, kernel_h * sizeof(const float *));

	const float *ptr_src = in_tensor.ptr_read();
	const float *ptr_bias = bias.ptr_read();
	for (int i_src = 0; i_src < src_shape[0]; i_src++)
	{
		for (int j_src = 0; j_src < src_shape[1]; j_src++)
		{
			for (int j_kernel = 0; j_kernel < kernel_w; j_kernel++)
			{
				for (int c_dst = 0; c_dst < kernel_c; c_dst++)
				{
					for (int c_src = 0; c_src < src_shape[2]; c_src++)
					{
						for (int i_kernel = 0; i_kernel < kernel_h; i_kernel++)
						{
							dst_ptrs[i_kernel][c_dst] += *(kernel_ptrs[i_kernel]++) * ptr_src[c_src];
						}
					}
				}
				// move pointer
				for (int i_kernel = 0; i_kernel < kernel_h; i_kernel++)
				{
					dst_ptrs[i_kernel] += kernel_c;
				}
			}
			// reset kernel pointer 
			memcpy(kernel_ptrs, kernel_ptrs_bak, kernel_h * sizeof(const float *));
			// move pointer to next dst pixel
			for (int i_kernel = 0; i_kernel < kernel_h; i_kernel++)
			{
				dst_ptrs[i_kernel] += dst_step_x;
			}
			ptr_src+= src_shape[2];
		}
		for (int i_kernel = 0; i_kernel < kernel_h; i_kernel++)
		{
			dst_ptrs[i_kernel] += dst_step_y;
		}
	}
	// add bias
	int dst_pix_amount = dst_shape[0] * dst_shape[1];
	float *dst_ptr = out_tensor.ptr_write();
	for (int i = 0; i < dst_pix_amount; i++)
	{
		for (int c = 0; c < dst_shape[2]; c++)
		{
			*(dst_ptr++) += ptr_bias[c];
		}
	}

	delete kernel_ptrs_release;
	delete kernel_ptrs_bak;
	delete dst_ptrs_release;
	out_ts = out_tensor;
	return void();
}

void antdnn::Cropping2D(Tensor & in_tensor, Tensor & out_ts, int top, int bottom, int left, int right)
{
	int *src_shape = in_tensor.shape();
	int dst_shape[3] = {
		src_shape[0] - top - bottom,
		src_shape[1] - left - right,
		src_shape[2]
	};
	Tensor out_tensor(3, dst_shape);
	auto src_ptr = in_tensor.ptr_read() + (top * src_shape[1] + left) * src_shape[2];
	auto dst_ptr = out_tensor.ptr_write();
	int src_rows = src_shape[1] * src_shape[2];
	int dst_rows = dst_shape[1] * dst_shape[2];
	for (int i = 0; i < dst_shape[0]; i++)
	{
		memcpy(dst_ptr, src_ptr, dst_rows);
		dst_ptr += dst_rows;
		src_ptr += src_rows;
	}
	out_ts = out_tensor;
	return void();
}
