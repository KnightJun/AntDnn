#include "Merge.h"
#include <assert.h>
DLL_ANTDNN_API void antdnn::add(int ts_amount, Tensor ** addvec, Tensor &ts_out)
{
	int ts_size = addvec[0]->size();
    for (int i = 1; i < ts_amount; i++)
	{
		assert(addvec[i]->size() == ts_size);
	}
	Tensor out_tensor(3, addvec[0]->shape());
	out_tensor.set_to(0);
	const float** src_ptrs = new const float*[ts_amount];
    for (int i = 0; i < ts_amount; i++)
	{
		src_ptrs[i] = addvec[i]->ptr_read();
	}
	float *dst_ptr = out_tensor.ptr_write();
    for (int i = 0; i < ts_size; i++)
	{
        for (int j = 0; j < ts_amount; j++)
		{
			*dst_ptr += *(src_ptrs[j]++);
		}
		dst_ptr++;
	}
	delete[] src_ptrs;
	ts_out = out_tensor;
	return void();
}
