#pragma once
#include "antdnn_macro.h"

#include <iostream>
#include <string>
#include <vector>
namespace antdnn
{
	struct size2D
	{
		int h = 1;
		int w = 1;
	};
	
	class DLL_ANTDNN_API Tensor
	{
	public:
		Tensor();
		Tensor(const int dim,const int *shape, float *src_data = nullptr);
		Tensor(const Tensor &ts);

		inline int dim() const { return m_dim; }
		inline const int* shape() const { return m_shape; }
		inline int size() const { return m_size; }
		inline int quote_count() const { return *m_quote_count; }
		void reshape(int dim, int *shape);

		void set_to(float val);
		inline const float* ptr_read() const { return m_data; };
		float *ptr_write();
		bool load_file(const std::string & filename);
		bool load_file(std::ifstream & infile);
		void recreate(const int dim, const int* shape, float * src_data = nullptr);

		std::string *errmsg = nullptr;
		Tensor & operator = (const Tensor &other);
		~Tensor();

		float sum();
		void makeBorder2D(int top, int bottom, int left, int right, const float c);

	private:

		int m_dim = 0;
		int m_size = 0;
		int *m_shape = nullptr;
		int *m_quote_count = nullptr;
		float *m_data = nullptr;
		void self_copy();
		void shallow_copy(const Tensor &ts);
		void release();

	};

	DLL_ANTDNN_API void load_tensor_vector(
		const std::string & filename,
		std::vector<Tensor> &tsvec);
	DLL_ANTDNN_API std::ostream& operator << (std::ostream& ost, antdnn::Tensor& ts);
}
