#include "Convolution.h"
#define forrange(i, d) for(int i = 0; i < d; i++)
#include <assert.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
int cpu_num = 4;
inline void o_mul_m_add(const float mul_1, const float* mul2, float *res, int count)
{
#ifdef USE_optimization_AVX
	if (count < 8)
	{
		for (int c_dst = 0; c_dst < count; c_dst++) // 32
		{
			res[c_dst] += mul_1 * mul2[c_dst];
		}
	}
	else
	{
		size_t cntBlock = count / 8;    // blocks
		size_t cntRem = count % 8;
		__m256 avx_mul_1 = _mm256_set1_ps(mul_1);
		for (int i = 0; i < cntBlock; ++i)
		{
			auto midres = _mm256_mul_ps(avx_mul_1, *(__m256 *)mul2);
			*(__m256 *)res = _mm256_add_ps(midres, *(__m256 *)res);
			res += 8;
			mul2 += 8;
		}
		for (int i = 0; i < cntRem; i++)
		{
			res[i] += mul_1 * mul2[i];
		}
	}
#else
	for (int c_dst = 0; c_dst < count; c_dst++) // 32
	{
		res[c_dst] += mul_1 * mul2[c_dst];
	}
#endif
}

void Conv2D_unit(const float *srcptr, const float *kerptr, float *dstptr, const float *ptr_bias,
	int calc_rows, int *srcshape,
	int *dst_shape,
	int *ker_shape)
{
	int old_rows = srcshape[1] * srcshape[2];
	int kernel_rows = ker_shape[1] * srcshape[2] * dst_shape[2];
	const float **rptrs_src = new const float*[ker_shape[0]];
	const float **rptrs_kernel = new const float*[ker_shape[0]];
	auto rptrs_src_release = rptrs_src, rptrs_kernel_release = rptrs_kernel;

	for (int i = 0; i < calc_rows; i++)
	{
		// set src rows ptr
		rptrs_src[0] = srcptr + i * old_rows;
		for (int i = 1; i < ker_shape[0]; i++)
		{
			rptrs_src[i] = rptrs_src[i - 1] + old_rows;
		}

		for (int j = 0; j < dst_shape[1]; j++)
		{
			// reset kernel row point
			rptrs_kernel[0] = kerptr;
			for (int i_r = 1; i_r < ker_shape[0]; i_r++)
			{
				rptrs_kernel[i_r] = rptrs_kernel[i_r - 1] + kernel_rows;
			}

			for (int j_kernel = 0; j_kernel < ker_shape[1]; j_kernel++) // 3
			{
				for (int c_src = 0; c_src < srcshape[2]; c_src++) // 3
				{
					//for (int c_dst = 0; c_dst < dst_shape[2]; c_dst++) // 32
					//{
					for (int i_kernel = 0; i_kernel < ker_shape[0]; i_kernel++) // 3
					{
						//dstptr[c_dst] += rptrs_src[i_kernel][c_src] * (*(rptrs_kernel[i_kernel]++));
						o_mul_m_add(rptrs_src[i_kernel][c_src], rptrs_kernel[i_kernel], dstptr, dst_shape[2]);
						rptrs_kernel[i_kernel] += dst_shape[2];
					}
					//}

				}
				for (int i_r = 0; i_r < ker_shape[0]; i_r++)
				{
					rptrs_src[i_r] += srcshape[2];
				}
			}
			//add bias
			for (int c_dst = 0; c_dst < dst_shape[2]; c_dst++) // 32
			{
				// output matrix next data
				*(dstptr++) += ptr_bias[c_dst];
			}
			// turn back the rows point 
			for (int i_r = 0; i_r < ker_shape[0]; i_r++)
			{
				rptrs_src[i_r] -= srcshape[2] * (ker_shape[1] - 1);
			}
		}
	}
	delete[] rptrs_src_release;
	delete[] rptrs_kernel_release;
}


void antdnn::Conv2D(
	Tensor & in_tensor, Tensor & out_ts,
	Tensor &weights, Tensor &bias,
	int padding_type)
{
	auto wei_shape = weights.shape();
	Tensor mid_tensor = in_tensor, out_tensor;
	if (padding_type == PADDING_SAME)
	{
		int top = (wei_shape[0] - 1) / 2,
			bottom = top,
			left = (wei_shape[1] - 1) / 2,
			right = left;
		mid_tensor.makeBorder2D(top, bottom, left, right, 0);
	}
	auto src_shape = mid_tensor.shape();
	int dst_shape[3] = {
		src_shape[0] - (wei_shape[0] - 1),
		src_shape[1] - (wei_shape[1] - 1) ,
		wei_shape[3] };
	out_tensor.recreate(3, dst_shape);
	out_tensor.set_to(0);
	// 3, 3, 3, 32
	float *dstptr = out_tensor.ptr_write();
	const float *srcptr = mid_tensor.ptr_read();
	auto kerptr = weights.ptr_read();
	auto biasptr = bias.ptr_read();

#ifdef USE_optimization_OPENMP
	int each_cpu_rows = dst_shape[0] / cpu_num;
	int last_rows = each_cpu_rows + dst_shape[0] % cpu_num;
	int each_cpu_rows_src = each_cpu_rows * src_shape[1] * src_shape[2];
	int each_cpu_rows_dst = each_cpu_rows * dst_shape[1] * dst_shape[2];
#pragma omp parallel for
	for (int i = 0; i < cpu_num; i++)
	{
		float *dstptr_omp = dstptr + each_cpu_rows_dst * i;
		const float *srcptr_omp = srcptr + each_cpu_rows_src * i;
		if (i < cpu_num - 1)
		{
			Conv2D_unit(srcptr_omp, kerptr, dstptr_omp, biasptr, each_cpu_rows,
				src_shape,
				dst_shape,
				wei_shape);
		}
		else
		{
			Conv2D_unit(srcptr_omp, kerptr, dstptr_omp, biasptr, last_rows,
				src_shape,
				dst_shape,
				wei_shape);
		}
	}
#else
	Conv2D_unit(srcptr, kerptr, dstptr, biasptr, dst_shape[0],
		src_shape,
		dst_shape,
		wei_shape);
#endif
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
	int dst_step_x = -(kernel_c * (kernel_w - strides_x));
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
			ptr_src += src_shape[2];
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
	int dst_rows_cpy = dst_rows * sizeof(float);
	for (int i = 0; i < dst_shape[0]; i++)
	{
		memcpy(dst_ptr, src_ptr, dst_rows_cpy);
		dst_ptr += dst_rows;
		src_ptr += src_rows;
	}
	out_ts = out_tensor;
	return void();
}