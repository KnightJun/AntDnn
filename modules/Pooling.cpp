#include "Pooling.h"

void antdnn::MaxPooling2D(Tensor & in_ts, Tensor & out_ts, int poolw, int poolh)
{
	size2D pool_size;
	pool_size.w = poolw;
	pool_size.h = poolh;
	int *inshape = in_ts.shape();
	int outshape[3] = {
		inshape[0] / pool_size.h,
		inshape[1] / pool_size.w,
		inshape[2]
	};
	int amend = inshape[1] - outshape[1] * pool_size.w;
	Tensor out_tensor(3, outshape);
	out_tensor.set_to(-3.40282e+038f);

	int old_row = inshape[1] * inshape[2];
	int new_row = outshape[1] * inshape[2];
	int channel = inshape[2];
	const float *ptr_src = in_ts.ptr_read();
	float *ptr_dst = out_tensor.ptr_write();
	for (int i_out = 0; i_out<outshape[0]; i_out++)
	{
		for (int i_pool = 0; i_pool < pool_size.h; i_pool++)
		{
			for (int j_out = 0; j_out < outshape[1]; j_out++)
			{
				for (int j_pool = 0; j_pool < pool_size.w; j_pool++)
				{
					for (int c = 0; c < channel; c++)
					{
						ptr_dst[c] = ptr_src[c] > ptr_dst[c] ? ptr_src[c] : ptr_dst[c];
					}
					ptr_src += channel;
				}
				ptr_dst += channel;
			}
			ptr_src += channel * amend;
			ptr_dst -= new_row;
		}
		ptr_dst += new_row;
	}
	out_ts = out_tensor;
	return void();
}
