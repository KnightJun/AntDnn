#include <antdnn/Tensor.h>
#include <antdnn/Convolution.h>
#include <antdnn/Activation.h>
#include <antdnn/Pooling.h>
#include <antdnn/Dense.h>
#include <antdnn/Merge.h>
#include <antdnn/TensorOperating.h>
#include <antdnn/RunTimeDetect.h>
#include <vector>
#include <iostream>
#include <time.h>
using namespace antdnn;
using namespace std;


void Conv2D_matmul(
	Tensor & in_tensor, Tensor & out_ts,
	Tensor &weights, Tensor &bias,
	int padding_type = PADDING_VALID)
{
	Tensor in_convmat;
	auto wei_shape = weights.shape();
	auto str_shape = in_tensor.shape();
	int dst_shape[3];
	if (padding_type == PADDING_SAME)
	{
		Tensor2ConvMat(in_tensor, in_convmat, 
			wei_shape[0], wei_shape[1],
			(wei_shape[0] - 1) / 2, (wei_shape[1] - 1) / 2);
		dst_shape[0] = str_shape[0];
		dst_shape[1] = str_shape[1];
		dst_shape[2] = wei_shape[3];
	}
	else
	{
		Tensor2ConvMat(in_tensor, in_convmat,
			wei_shape[0], wei_shape[1],
			0, 0);
		dst_shape[0] = str_shape[0] - (wei_shape[0] - 1);
		dst_shape[1] = str_shape[1] - (wei_shape[1] - 1);
		dst_shape[2] = wei_shape[3];
	}
    out_ts.recreate(3, dst_shape);
	Matrix_Mul(in_convmat, weights, out_ts);

	auto resptr = out_ts.ptr_write();
	auto weiptr = bias.ptr_read();
	auto channels = *bias.shape();
	auto pix_count = dst_shape[0]* dst_shape[1];
	for (int i = 0; i < pix_count; i++)
	{
		for (int c = 0; c < channels; c++)
		{
			*resptr = *resptr + weiptr[c];
			resptr++;
		}
	}
	return;
}
void TEST_fcn()
{
#define Conv2D Conv2D_matmul
	vector<Tensor> tsvec(38);
	Tensor in_ts, out_ts;
	tsvec.resize(10);
    load_tensor_vector("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\fcn_weights.antlist", tsvec);
    in_ts.load_file("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\fcn_img.antts");
	
	clock_t start = clock();
	Conv2D(in_ts, out_ts, tsvec[0], tsvec[1], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[2], tsvec[3], PADDING_SAME); activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);

	Conv2D(out_ts, out_ts, tsvec[4], tsvec[5], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[6], tsvec[7], PADDING_SAME); activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);

	Conv2D(out_ts, out_ts, tsvec[8], tsvec[9], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[10], tsvec[11], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[12], tsvec[13], PADDING_SAME); activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);
	auto layers_p3 = out_ts; // split layer

	Conv2D(out_ts, out_ts, tsvec[14], tsvec[15], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[16], tsvec[17], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[18], tsvec[19], PADDING_SAME); activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);

	auto layers_p4 = out_ts; // split layer
	Conv2D(out_ts, out_ts, tsvec[20], tsvec[21], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[22], tsvec[23], PADDING_SAME); activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[24], tsvec[25], PADDING_SAME); activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);
	auto layers_p5 = out_ts; // split layer
//#undef Conv2D
	Conv2D(layers_p4, layers_p4, tsvec[26], tsvec[27]); activation_relu(layers_p4);
	Conv2D(layers_p5, layers_p5, tsvec[28], tsvec[29]); activation_relu(layers_p5);
	Conv2DTranspose(layers_p4, layers_p4, tsvec[30], tsvec[31], PADDING_VALID, 2, 2); // 30, 30, 4
	Conv2DTranspose(layers_p5, layers_p5, tsvec[32], tsvec[33], PADDING_VALID, 4, 4); // 30, 30, 4
	Conv2D(layers_p3, layers_p3, tsvec[34], tsvec[35]); activation_relu(layers_p3);
	
	Cropping2D(layers_p4, layers_p4, 1, 1, 1, 1);
	Cropping2D(layers_p5, layers_p5, 2, 2, 2, 2);

	Tensor *addvec[3] = { &layers_p3, &layers_p4, &layers_p5 };
	add(3, addvec, out_ts);
	Conv2DTranspose(out_ts, out_ts, tsvec[36], tsvec[37], PADDING_VALID, 8, 8);
	Cropping2D(out_ts, out_ts, 4, 4, 4, 4);
	activation_softmax(out_ts);

	clock_t ends = clock();
	float res = out_ts.sum();
	if (0x4743fece == *(int *)&res)
	{
		cout << "successful" << endl;
	}
	else
	{
		cout << "false" << endl;
	}
	cout << "Running Time : " << (double)(ends - start) / CLOCKS_PER_SEC << endl;
	cout << out_ts.shape()[0] << "," << out_ts.shape()[1] << "," << out_ts.shape()[2] << endl;
	cout << out_ts.sum() << endl;
	PrintRunTime();
}


int main()
{
	for (size_t i = 0; i < 1; i++)
	{
		TEST_fcn();
    }
	return 0;
}
