## AntDnn 

​	AntDnn is a C++ libary for machine learning network forward calculation. It don't depend on any libary. It is easy  to embed into most program, system, embedded device.

#### feature

- train by keras(with tensorflow backend), predict by antdnn.

#### Supported layers (name from keras)

- Conv2D
- Conv2DTranspose
- Cropping2D
- Dense
- add
- MaxPooling2D



### Using 

#### main data type: Tensor

​	Multi-dimension float data, it was the only data type in antdnn, represent weights, input or output data.

#### CNN example 

> Training code:https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

``` c++
	Tensor in_ts, out_ts;
	vector<Tensor> tsvec;
	// replace the path to your path
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
```



​	