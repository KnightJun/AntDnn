#include "Pooling.h"
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
extern int cpu_num;

inline void m_max(const float* ptr_src, float *ptr_dst, int size)
{
#if USE_optimization_AVX==1
	if (size < 8)
	{
		for (int c = 0; c < size; c++) // 32
		{
			ptr_dst[c] = ptr_src[c] > ptr_dst[c] ? ptr_src[c] : ptr_dst[c];
		}
	}
	else
	{
		size_t cntBlock = size / 8;    // blocks
		size_t cntRem = size % 8;
		for (int i = 0; i < cntBlock; ++i)
		{
			*(__m256 *)ptr_dst = _mm256_max_ps(*(__m256 *)ptr_src, *(__m256 *)ptr_dst);
			ptr_dst += 8;
			ptr_src += 8;
		}
		for (int i = 0; i < cntRem; i++)
		{
			ptr_dst[i] = ptr_src[i] > ptr_dst[i] ? ptr_src[i] : ptr_dst[i];
		}
	}
#else
	for (int c = 0; c < size; c++)
	{
		ptr_dst[c] = ptr_src[c] > ptr_dst[c] ? ptr_src[c] : ptr_dst[c];
	}
#endif
}
void MaxPooling2D_unit(const float* ptr_src, float *ptr_dst,
	int pool_size_h, int pool_size_w, int amend, int new_row,
	int calcrow, const int *inshape, const int *outshape)
{
	for (int i_out = 0; i_out < calcrow; i_out++)
	{
		for (int i_pool = 0; i_pool < pool_size_h; i_pool++)
		{
			for (int j_out = 0; j_out < outshape[1]; j_out++)
			{
				for (int j_pool = 0; j_pool < pool_size_w; j_pool++)
				{
					/*for (int c = 0; c < inshape[2]; c++)
					{
						ptr_dst[c] = ptr_src[c] > ptr_dst[c] ? ptr_src[c] : ptr_dst[c];
					}*/
					m_max(ptr_src, ptr_dst, inshape[2]);
					ptr_src += inshape[2];
				}
				ptr_dst += inshape[2];
			}
			ptr_src += inshape[2] * amend;
			ptr_dst -= new_row;
		}
		ptr_dst += new_row;
	}
}
void antdnn::MaxPooling2D(Tensor & in_ts, Tensor & out_ts, int poolw, int poolh)
{
	const int *inshape = in_ts.shape();
	int outshape[3] = {
		inshape[0] / poolh,
		inshape[1] / poolw,
		inshape[2]
	};
	int amend = inshape[1] - outshape[1] * poolw;
	Tensor out_tensor(3, outshape);
	out_tensor.set_to(-3.40282e+038f);

    int new_row = outshape[1] * inshape[2];
	const float *ptr_src = in_ts.ptr_read();
	float *ptr_dst = out_tensor.ptr_write();
#if 1 //USE_optimization_OPENMP
	MaxPooling2D_unit(ptr_src, ptr_dst, poolh, poolw, amend,
		new_row, outshape[0], inshape, outshape);
#else
	int each_cpu_rows = outshape[0] / cpu_num;
	int last_rows = each_cpu_rows + outshape[0] % cpu_num;
	int each_cpu_rows_src = each_cpu_rows * inshape[1] * inshape[2] * poolh;
	int each_cpu_rows_dst = each_cpu_rows * outshape[1] * outshape[2];
#pragma omp parallel for
	for (int i = 0; i < cpu_num; i++)
	{
		float *dstptr_omp = ptr_dst + each_cpu_rows_dst * i;
		const float *srcptr_omp = ptr_src + each_cpu_rows_src * i;
		if (i < cpu_num - 1) {
			MaxPooling2D_unit(srcptr_omp, dstptr_omp, poolh, poolw, amend,
				new_row, each_cpu_rows, inshape, outshape);
		}
		else
		{
			MaxPooling2D_unit(srcptr_omp, dstptr_omp, poolh, poolw, amend,
				new_row, last_rows, inshape, outshape);
		}
	}
#endif
	out_ts = out_tensor;
	return void();
}
