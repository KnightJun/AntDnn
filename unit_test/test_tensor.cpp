#include <gtest/gtest.h>
#include <antdnn/Tensor.h>
#include <antdnn/Convolution.h>
#include <antdnn/Activation.h>
#include <antdnn/Pooling.h>
#include <antdnn/Dense.h>
#include <antdnn/Merge.h>
#include <vector>
#include <iostream>
using namespace antdnn;
using namespace std;
TEST(tensor, create)
{
	auto ts = new Tensor;
	Tensor ts2 = *ts;
	EXPECT_EQ(ts->dim() , 0);
	EXPECT_EQ(ts->ptr_read(), ts2.ptr_read());
	EXPECT_TRUE(ts2.ptr_write() == nullptr && ts->ptr_read() == nullptr);
	delete ts;

	int shape[] = { 12, 11, 7, 3 };
	auto ts3 = new Tensor(4, shape);
	EXPECT_EQ(ts3->dim(), 4);
	EXPECT_EQ(ts3->size(), 12 * 11 * 7 * 3);
	ts3->set_to(421);
	auto ptr = ts3->ptr_read();
	bool res = true;
	for (int i = 0; i < ts3->size(); i++)
	{
		if (ptr[i] != 421)
			res = false;
	}
	EXPECT_TRUE(res);

	Tensor ts4 = *ts3;
	EXPECT_EQ(ts3->ptr_read(), ts4.ptr_read());
	EXPECT_EQ(ts4.quote_count(), 2);
	ts4.set_to(123);
	EXPECT_NE(ts3->ptr_read(), ts4.ptr_read());
	EXPECT_EQ(ts4.quote_count(), 1);
	EXPECT_EQ(ts3->quote_count(), 1);
	EXPECT_EQ(*ts3->ptr_read(), 421);
	EXPECT_EQ(*ts4.ptr_read(), 123);
	delete ts3;
	ts4.recreate(3, shape);
	EXPECT_EQ(ts4.size(), 12 * 11 * 7);
}

TEST(tensor, makebounder)
{
	int dim[] = {3, 3, 3};
	Tensor ts = Tensor(3, dim);
	ts.set_to(1);
	ts.makeBorder2D(1, 1, 1, 1, 2.0);
	// std::cout << ts;
	EXPECT_EQ(ts.sum(), 27 + ((5 + 3 + 3 + 5) * 2)*3);
}

TEST(tensor, conv2d)
{
	
	Tensor in_ts, out_ts;
	vector<Tensor> tsvec;

	load_tensor_vector("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\weights.antlist", tsvec);
	in_ts.load_file("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\img.antts");

	Conv2D(in_ts, out_ts, tsvec[0], tsvec[1], PADDING_SAME);
	activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[2], tsvec[3]);
	activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2,2);

	Conv2D(out_ts, out_ts, tsvec[4], tsvec[5], PADDING_SAME);
	activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[6], tsvec[7]);
	activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2,2);

	flatten(out_ts);
	Dense(out_ts, out_ts, tsvec[8], tsvec[9]);
	activation_relu(out_ts);
	Dense(out_ts, out_ts, tsvec[10], tsvec[11]);
	activation_softmax(out_ts);
	cout << out_ts << endl;
}

TEST(tensor, fcn)
{
	vector<Tensor> tsvec;
	Tensor in_ts, out_ts;

	load_tensor_vector("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\fcn_weights.antlist", tsvec);
	in_ts.load_file("C:\\Users\\jun\\OneDrive\\code\\cppdnn\\python\\fcn_img.antts");
	auto it_wei = tsvec.begin();
	Conv2D(in_ts, out_ts, tsvec[0], tsvec[1], PADDING_SAME);activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[2], tsvec[3], PADDING_SAME);activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);

	Conv2D(out_ts, out_ts, tsvec[4], tsvec[5], PADDING_SAME);activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[6], tsvec[7], PADDING_SAME);activation_relu(out_ts);
	MaxPooling2D(out_ts, out_ts, 2, 2);

	Conv2D(out_ts, out_ts, tsvec[8], tsvec[9], PADDING_SAME);activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[10], tsvec[11], PADDING_SAME);activation_relu(out_ts);
	Conv2D(out_ts, out_ts, tsvec[12], tsvec[13], PADDING_SAME);activation_relu(out_ts);
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

	Conv2D(layers_p4,layers_p4, tsvec[26], tsvec[27]); activation_relu(layers_p4);
	Conv2D(layers_p5, layers_p5, tsvec[28], tsvec[29]); activation_relu(layers_p5);
	Conv2DTranspose(layers_p4, layers_p4, tsvec[30], tsvec[31], PADDING_VALID, 2, 2); // 30, 30, 4
	Conv2DTranspose(layers_p5, layers_p5, tsvec[32], tsvec[33], PADDING_VALID, 4, 4); // 30, 30, 4
	Conv2D(layers_p3, layers_p3, tsvec[34], tsvec[35]); activation_relu(layers_p3);

	Cropping2D(layers_p4, layers_p4, 1, 1, 1, 1);
	Cropping2D(layers_p5, layers_p5, 2, 2, 2, 2);

	Tensor *addvec[3] = { &layers_p3, &layers_p4, &layers_p5};
	add(3, addvec, out_ts);
	Conv2DTranspose(out_ts, out_ts, tsvec[36], tsvec[37], PADDING_VALID, 8, 8);
	Cropping2D(out_ts, out_ts, 4, 4, 4, 4);
	activation_softmax(out_ts);

	cout << out_ts.shape()[0] << "," << out_ts.shape()[1] << "," << out_ts.shape()[2] << endl;
	cout << out_ts.sum() << endl;
}
int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);

	// Runs all tests using Google Test. 
	return RUN_ALL_TESTS();
}