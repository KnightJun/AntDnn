#include "Activation.h"
#include <math.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
void antdnn::activation_relu(Tensor & ts)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();

#ifdef USE_optimization_AVX
	if (tsize < 8)
	{
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < tsize; i++)
			ptrw[i] = __max(ptrw[i], 0);
	}
	else
	{
		size_t cntBlock = tsize / 8;    // blocks
		size_t cntRem = tsize % 8;
		__m256 avx_zeros = _mm256_setzero_ps();
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < cntBlock; i++)
		{
			auto tmpptr = ptrw + i * 8;
			*(__m256*)(tmpptr) = _mm256_max_ps(avx_zeros, *(__m256*)tmpptr);
		}
		for (int i = tsize - cntRem; i < tsize; i++)
		{
			ptrw[i] = __max(ptrw[i], 0);
		}
	}
		
#else
#ifdef USE_optimization_OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < tsize; i++)
		ptrw[i] = __max(ptrw[i], 0);
#endif
}

void antdnn::activation_softmax(Tensor & ts)
{
	auto ptrw = ts.ptr_write();
	int size_channels = ts.shape()[ts.dim() - 1];
	int size_pixel = ts.size() / size_channels;
	float sum;
	for (size_t i = 0; i < size_pixel; i++)
	{
		sum = 0;
		for (int c = 0; c < size_channels; c++)
		{
			sum += expf(ptrw[c]);
		}
		for (int c = 0; c < size_channels; c++)
		{
			ptrw[c] = expf(ptrw[c]) / sum;
		}
		ptrw += size_channels;
	}
	return void();
}

void antdnn::flatten(Tensor & ts)
{
	int size = ts.size();
	ts.reshape(1, &size);
	return void();
}
