#include "..\include\antdnn\Dense.h"

void antdnn::Dense(Tensor & in_ts, Tensor & out_ts, Tensor & weights, Tensor & bias)
{
	int new_size = bias.size();
	int old_size = in_ts.size();
	Tensor out_tensor(1, &new_size);
	const float *ptr_old = in_ts.ptr_read();
	const float *ptr_wei = weights.ptr_read();
	const float *ptr_bias = bias.ptr_read();
	float *ptr_new = out_tensor.ptr_write();
	out_tensor.set_to(0);
	for (size_t i_old = 0; i_old < old_size; i_old++)
	{
		for (size_t i_new = 0; i_new < new_size; i_new++)
		{
			ptr_new[i_new] += *ptr_old * *(ptr_wei++);
		}
		ptr_old++;
	}
	for (size_t i_new = 0; i_new < new_size; i_new++)
	{
		ptr_new[i_new] += ptr_bias[i_new];
	}
	out_ts = out_tensor;
	return void();
}
