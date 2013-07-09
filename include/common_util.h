/**
 * Common non-opencl code used by dwarves & test code
 */

#include<stdlib.h>
#include<stdio.h>

#include<config.h>

#define MINIMUM(i,j) ((i)<(j) ? (i) : (j))
#define ACL_ALIGNMENT 64 // Minimum alignment for DMA transfer to Altera FPGA board

extern void check();

extern void* int_new_array(const size_t N,const char* error_msg);
extern void* long_new_array(const size_t N,const char* error_msg);
extern void* float_new_array(const size_t N,const char* error_msg);
