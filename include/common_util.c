#include "common_util.h"

void check(int b,const char* msg)
{
	if(!b)
	{
		fprintf(stderr,"error: %s\n\n",msg);
		exit(-1);
	}
}

void* char_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	#ifdef USE_AFPGA
		err = posix_memalign(&ptr,ACL_ALIGNMENT,N * sizeof(char));
		check(err == 0,error_msg);
	#else
		ptr = malloc(N * sizeof(char));
		check(ptr != NULL,error_msg);
	#endif
	return ptr;
}

void* int_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	#ifdef USE_AFPGA
		err = posix_memalign(&ptr,ACL_ALIGNMENT,N * sizeof(int));
		check(err == 0,error_msg);
	#else
		ptr = malloc(N * sizeof(int));
		check(ptr != NULL,error_msg);
	#endif
	return ptr;
}

void* long_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	#ifdef USE_AFPGA
		err = posix_memalign(&ptr,ACL_ALIGNMENT,N * sizeof(long));
		check(err == 0,error_msg);
	#else
		ptr = malloc(N * sizeof(long));
		check(ptr != NULL,error_msg);
	#endif
	return ptr;
}

void* float_new_array(const size_t N,const char* error_msg)
{
	void* ptr;
	int err;
	#ifdef USE_AFPGA
		err = posix_memalign(&ptr,ACL_ALIGNMENT,N * sizeof(float));
		check(!err,error_msg);
	#else
		ptr = malloc(N * sizeof(float));
		check(ptr != NULL,error_msg);
	#endif
	return ptr;
}

void* float_array_realloc(void* ptr,const size_t N,const char* error_msg)
{
	int err;
	#ifdef USE_AFPGA
		if(ptr != NULL) free(ptr);
		err = posix_memalign(&ptr,ACL_ALIGNMENT,N * sizeof(float));
		check(!err,error_msg);
	#else
		ptr = realloc(ptr,N * sizeof(float));
		check(ptr != NULL,error_msg);
	#endif
	return ptr;
}
