#ifndef CRC_FORMATS_H
#define CRC_FORMATS_H

#include<stdio.h>
#include<stdlib.h>

#include "../../../include/common_util.h"

unsigned char** read_crc(unsigned int* num_pages,unsigned int* page_size,const char* file_path);
void write_crc(const unsigned char** pages, const unsigned int num_pages, const unsigned int page_size,const char* file_path);
unsigned char** rand_crc(const unsigned int num_pages,const unsigned int page_size,const unsigned int seed);
void free_crc(unsigned char** pages, const unsigned int num_pages);


#endif
