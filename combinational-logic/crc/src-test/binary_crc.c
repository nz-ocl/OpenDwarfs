#include "../../../include/common_util.h"
#include "../inc/crc_formats.h"

#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<unistd.h>

int main(int argc, char** argv)
{
  unsigned int page_size,num_pages;
  unsigned char** pages;
  FILE* fp;
  size_t write_count;

  pages = read_crc(&num_pages,&page_size,"../test/combinational-logic/crc/crcfile_N1_S10");

  fp = fopen("../test/combinational-logic/crc/binary-crcfile_N1_S10","w");
  check(fp != NULL,"Cannot open output file!");

  write_count = fwrite(pages[0],sizeof(char),page_size,fp);
  check(write_count == page_size,"write_count != page size!");


  free_crc(pages,num_pages);
  return 0;
}
