#pragma once
#define USE_optimization_OPENMP
#define USE_optimization_AVX
#ifdef _MSC_VER
	#ifdef _DLL_ANTDNN
		#define DLL_ANTDNN_API __declspec(dllexport)
	#else  
		#define DLL_ANTDNN_API __declspec(dllimport)
	#endif  
#else
	#define DLL_ANTDNN_API
#endif // _MSC_VER

