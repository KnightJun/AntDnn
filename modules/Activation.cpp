#include "Activation.h"
#include <math.h>
#define __max(a,b)  (((a) > (b)) ? (a) : (b))
void antdnn::activation_relu(Tensor & ts)
{
	int tsize = ts.size();
	auto ptrw = ts.ptr_write();
	for (int i = 0; i < tsize; i++)
	{
		ptrw[i] = __max(ptrw[i], 0);
	}
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
