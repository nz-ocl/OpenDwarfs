#!/bin/sh

##SCP files to ocldev for execution
USER=tjkenney
OCLDEV_IP=9.31.103.247

#########SPMV###########

##Source
scp -r /home/tjkenney/OpenDwarfs/sparse-linear-algebra/SPMV $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/sparse-linear-algebra/

##Fpga-exec
scp /home/tjkenney/OpenDwarfs/fpga-build/csr $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/fpga-exec/
#scp tjkenney@fpgabuildrhl.nzlab.ibm.com:/home/tjkenney/open_dwarfs/altera/spmv/spmv_csr_kernel.aocx tjkenney@9.31.103.89:/home/tjkenney/open_dwarfs/fpga-exec/

##xpu-exec
scp /home/tjkenney/OpenDwarfs/xpu-build/csr $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/xpu-exec/
#scp /home/tjkenney/OpenDwarfs/xpu-build/spmv_csr_kernel.cl tjkenney@9.31.103.89:/home/tjkenney/open_dwarfs/xpu-exec/

########SPMV###########

########CRC###########

##Source
scp -r /home/tjkenney/OpenDwarfs/combinational-logic/crc/ $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/combinational-logic/

##Fpga-exec
scp /home/tjkenney/OpenDwarfs/fpga-build/crc $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/fpga-exec/

##xpu-exec
scp /home/tjkenney/OpenDwarfs/xpu-build/crc $USER@$OCLDEV_IP:/home/tjkenney/open_dwarfs/xpu-exec/

#########CRC###########
