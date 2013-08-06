#include "/home/tjkenney/OpenDwarfs/combinational-logic/crc/inc/eth_crc32_lut.h"

__kernel void crc32_slice8(__global uint* data, uint length,__global uint* res)
{
  uint crc = 0xFFFFFFFF;
  unsigned char* currentChar;
  __private uint one,two;
  size_t i=0,j=0;
  // process eight bytes at once
  while (length >= 8)
  {
    one = data[i++] ^ crc;
    two = data[i++];
    crc = crc32Lookup[7][ one      & 0xFF] ^
          crc32Lookup[6][(one>> 8) & 0xFF] ^
          crc32Lookup[5][(one>>16) & 0xFF] ^
          crc32Lookup[4][ one>>24        ] ^
          crc32Lookup[3][ two      & 0xFF] ^
          crc32Lookup[2][(two>> 8) & 0xFF] ^
          crc32Lookup[1][(two>>16) & 0xFF] ^
          crc32Lookup[0][ two>>24        ];
    length -= 8;
  }
  
  // remaining 1 to 7 bytes
  while(length)
  {
	  one = data[i++];
	 currentChar = (unsigned char*) &one;
	  j=0;
	  while (length && j < 4) 
	  {
	  	length = length - 1;
	    crc = (crc >> 8) ^ crc32Lookup[0][(crc & 0xFF) ^ currentChar[j]];
	    j = j + 1;
	  }
  }
  res[0] = ~crc;
}