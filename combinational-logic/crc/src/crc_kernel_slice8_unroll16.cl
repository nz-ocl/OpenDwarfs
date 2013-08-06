#include "/home/tjkenney/OpenDwarfs/combinational-logic/crc/inc/eth_crc32_lut.h"

#define UNROLL_FACTOR 16

__kernel void crc32_slice8(__global uint* data, uint length,__global uint* partial_remainder)
{
  __private uint crc =  0xFFFFFFFF;
  __private unsigned char* currentChar;
  __private uint one,two;
  __private size_t i,ii,j,gid;
  
  gid = get_global_id(0);
  i = gid*UNROLL_FACTOR*2;
  
  #pragma unroll UNROLL_FACTOR
  for(ii=0; ii<UNROLL_FACTOR; ii++)
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
  }
  
  partial_remainder[gid] = ~crc;
}

__kernel void reduce(uint num_partial_remainders, __global uint* partial_remainder, __global uint* ret)
{
	__private uint i;
	__private uint crc = 0x00000000;
	
	for(i=0; i<num_partial_remainders; i++) {
			crc = crc ^ partial_remainder[i];
	}
	
	ret[0] = crc;
}