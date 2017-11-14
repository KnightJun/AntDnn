#include "Tensor.h"
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <assert.h>
#ifdef USE_optimization_OPENMP
#include <omp.h>
#endif
#define forrange(i, d) for(int i = 0; i < d; i++)
using namespace std;
const char* flag_file = "ant";
const char* flag_list = "lis";
const char* flag_tensor = "tsd";
int cpu_num = 0;
inline int calcSize(int dim, const int *shape)
{
	int size = 1;
	for (int i = 0; i < dim; i++)
	{
		size *= *(shape++);
	}
	return size;
}

antdnn::Tensor::Tensor()
{
#ifdef USE_optimization_OPENMP
	if (cpu_num == 0)  cpu_num = omp_get_num_procs();
#endif // USE_optimization_OPENMP

	errmsg = new string;
	m_quote_count = new int(1);
}

antdnn::Tensor::~Tensor()
{
	delete errmsg;
	release();
}
antdnn::Tensor::Tensor(const int dim, const int *shape, float * src_data)
{
#ifdef USE_optimization_OPENMP
	if (cpu_num == 0)  cpu_num = omp_get_num_procs();
#endif // USE_optimization_OPENMP
	errmsg = new string;
	this->recreate(dim, shape, src_data);
}

antdnn::Tensor::Tensor(const Tensor & ts)
{
	errmsg = new string;
	this->shallow_copy(ts);
}

void antdnn::Tensor::reshape(int dim, int * shape)
{
	int new_size = calcSize(dim, shape);
	assert(new_size == this->m_size);
	delete this->m_shape;
	this->m_dim = dim;
	this->m_shape = new int[dim];
	memcpy(this->m_shape, shape, dim * sizeof(int));
	return void();
}

void antdnn::Tensor::set_to(float val)
{
	this->self_copy();
	for (int i = 0; i < m_size; i++)
	{
		m_data[i] = val;
	}
}


float * antdnn::Tensor::ptr_write()
{
	if (this->m_dim > 0)
	{
		this->self_copy();
	}
	return m_data;
}

bool antdnn::Tensor::load_file(const std::string &filename)
{
	std::ifstream infile(filename, ios::in | ios::binary);
	int flat;
	infile.read((char *)&flat, 4);
	if (flat != *(int *)flag_file)
	{
		*errmsg = "not a antdnn data file.";
		infile.close();
		return false;
	}
	this->load_file(infile);
	infile.close();
	return true;
}

bool antdnn::Tensor::load_file(std::ifstream & infile)
{
	int flat;
	infile.read((char *)&flat, 4);
	if (flat != *(int *)flag_tensor)
	{
		*errmsg = "not a tensor file.";
		infile.close();
		return false;
	}
	this->release();
	infile.read((char *)&this->m_dim, 4);
	this->m_shape = new int[this->m_dim];
	infile.read((char *)this->m_shape, 4 * this->m_dim);
	this->m_size = calcSize(this->m_dim, this->m_shape);
	this->m_data = new float[this->m_size];
	this->m_quote_count = new int(1);
	infile.read((char *)this->m_data, this->m_size * sizeof(float));
	return true;
}

antdnn::Tensor & antdnn::Tensor::operator=(const Tensor & other)
{
	this->release();
	this->shallow_copy(other);
	return *this;
}



void antdnn::Tensor::self_copy()
{
	if (*m_quote_count == 1)
	{
		return;
	}
	// copy itself
	*m_quote_count -= 1;
	auto new_shape = new int[m_dim];
	memcpy(new_shape, m_shape, sizeof(int)*m_dim);
	auto new_q_count = new int(1);
	auto new_data = new float[m_size];
	memcpy(new_data, m_data, sizeof(float) * m_size);
	m_quote_count = new_q_count;
	m_shape = new_shape;
	m_data = new_data;
	return;
}

void antdnn::Tensor::shallow_copy(const Tensor & ts)
{
	if (ts.m_dim == 0)
	{
		this->release();
		return;
	}
	this->m_dim = ts.m_dim;
	this->m_quote_count = ts.m_quote_count;
	(*this->m_quote_count)++;
	this->m_data = ts.m_data;
	this->m_size = ts.m_size;
	if (m_dim > 0)
	{
		this->m_shape = new int[this->m_dim];
		memcpy(this->m_shape, ts.m_shape, sizeof(int) * this->m_dim);
	}
}

void antdnn::Tensor::recreate(const int dim, const int * shape, float * src_data)
{
	this->release();
	m_dim = dim;
	m_shape = new int[m_dim];
	memcpy(m_shape, shape, dim * sizeof(int));
	m_size = calcSize(m_dim, m_shape);
	m_data = new float[m_size];
	if (src_data != nullptr)
	{
		memcpy(m_data, src_data, m_size * sizeof(float));
	}
	m_quote_count = new int(1);
}

void antdnn::Tensor::release()
{
	if (m_quote_count == nullptr)
		return;
	*m_quote_count -= 1;
	if (*m_quote_count == 0)
	{
		delete m_quote_count;
		delete[] m_data;
	}
	delete[] m_shape;
	m_size = 0;
	m_dim = 0;
	m_quote_count = nullptr;
	m_data = nullptr;
	m_shape = nullptr;
}

float antdnn::Tensor::sum()
{
	float res = 0;
	for (int i = 0; i < this->m_size; i ++)
	{
		res += this->m_data[i];
	}
	return res;
}

void antdnn::Tensor::makeBorder2D(int top, int bottom, int left, int right, const float c)
{
	assert(this->m_dim == 3);
	int *new_shape = new int[3];
	new_shape[0] = this->m_shape[0] + top + bottom;
	new_shape[1] = this->m_shape[1] + left + bottom;
	new_shape[2] = this->m_shape[2];

	int new_size = calcSize(3, new_shape);
	auto old_data = this->m_data;
	float *new_data = new float[new_size];
	auto cpy_data = new_data;
	// fill the top part of new matrix
	for (int i = 0; i < top * new_shape[1] * new_shape[2]; i++) 
	{
		*(cpy_data++) = c;
	}
	int old_rows = this->m_shape[1] * this->m_shape[2];
	for (int i = 0; i < this->m_shape[0]; i++)
	{
		// fill the new left part
		for (int j = 0; j < left * new_shape[2]; j++)
		{
			*(cpy_data++) = c;
		}
		// copy middle part of source
		memcpy(cpy_data, old_data, old_rows * sizeof(float));
		// fill the new right part
		old_data += old_rows;
		cpy_data += old_rows;
		for (int j = 0; j < right * new_shape[2]; j++)
		{
			*(cpy_data++) = c;
		}
	}
	// fill the bottom part of new matrix
	for (int i = 0; i < bottom * new_shape[1] * new_shape[2]; i++) 
	{
		*(cpy_data++) = c;
	}
	this->release();
	this->m_dim = 3;
	this->m_shape = new_shape;
	this->m_quote_count = new int(1);
	this->m_data = new_data;
	this->m_size = new_size;
}

void _output_line(std::ostream & ost, int size,const float* &data)
{
	ost << "[";
	if (size > 6)
	{
		for (int i = 0; i < 3; i++)
		{
			ost << " " << *(data++) << ",";
		}
		ost << " ...,";
		data += size - 6;
		for (int i = size - 3; i < size - 1; i++)
		{
			ost << " " << *(data++) << ",";
		}
		ost << " " << *(data++);
	}
	else
	{
		for (int i = 0; i < size - 1; i++)
		{
			ost << " " << *(data++) << ",";
		}
		ost << " " << *(data++);
	}
	ost << "]";
}

void __output(std::ostream & ost, 
	int dim, 
	const int *shape,
	const float* &data, 
	int size)
{
	if (dim == 1)
	{
		ost << '[';
		_output_line(ost, shape[dim - 1], data);
		ost << ']' << std::endl;
		return;
	}
	int zero;
	auto mdim = dim - 1;
	int *count = new int[mdim];
	for (int i = 0; i < mdim; i++)count[i] = 0;
	for (int i_size = 0; i_size < size; i_size+= shape[mdim])
	{
		zero = 0;
		for (int i = mdim - 1; i >= 0; i--)
			if (count[i] == 0)zero++; else break;
		// output break and [
		for (int i = 0; i < mdim - zero; i++)ost << ' ';
		for (int i = 0; i < zero; i++)ost << '[';
		_output_line(ost, shape[dim - 1], data);
		count[mdim - 1]++;
		// add count
		int carry = 0;
		int jump_count = 0, csize = 0;
		for (int i = mdim - 1; i >= 0; i--)
		{

			count[i] += carry;
			if (shape[i] > 6 && count[i] == 3) // jump
			{
				jump_count = shape[i] - 6;
				csize = calcSize(dim - i - 1, shape + i + 1);
				count[i] += jump_count;
				data += csize * jump_count;
				i_size += csize * jump_count;
			}
			if (count[i] == shape[i])
			{
				count[i] = 0;
				carry = 1;
			}
			else
			{
				break;
			}
		}

		zero = 0;
		for (int i = mdim - 1; i >= 0; i--)
			if (count[i] == 0)zero++; else break;
		forrange(i, zero)ost << ']';
		if (i_size == size - shape[mdim])
		{
			ost << std::endl;
			break;
		}
		ost << ',';
		ost << std::endl;
		forrange(i, zero)ost << std::endl;
		if (jump_count > 0)
		{
			for (int i = 0; i < mdim - zero; i++)ost << ' ';
			ost << "...,";
			ost << std::endl;
		}
	}
	delete[] count;
}

void antdnn::load_tensor_vector(const std::string & filename, std::vector<antdnn::Tensor>& tsvec)
{
	std::ifstream infile(filename, ios::in | ios::binary);
	int flat;
	infile.read((char *)&flat, 4);
	if (flat != *(int *)flag_file)
	{
		return;
	}
	infile.read((char *)&flat, 4);
	if (flat != *(int *)flag_list)
	{
		return;
	}
	int listlen;
	infile.read((char *)&listlen, 4);
	tsvec.resize(listlen);
	for (auto &ts : tsvec)
	{
		ts.load_file(infile);
	}
	infile.close();
	return void();
}

std::ostream & antdnn::operator<<(std::ostream & ost, antdnn::Tensor & ts)
{
	auto bak_data = ts.ptr_read();;
	__output(ost, ts.dim(), ts.shape(), bak_data, ts.size());
	return ost;
	// TODO: 在此处插入 return 语句
}