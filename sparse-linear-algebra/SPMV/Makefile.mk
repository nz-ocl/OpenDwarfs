#
# Copyright 2010 by Virginia Polytechnic Institute and State
# University. All rights reserved. Virginia Polytechnic Institute and
# State University (Virginia Tech) owns the software and its
# associated documentation.
#

bin_PROGRAMS += csr
bin_PROGRAMS += createcsr


csr_SOURCES = sparse-linear-algebra/SPMV/src/csr.c
createcsr_SOURCES = sparse-linear-algebra/SPMV/test-src/createcsr.c 

##createcsr does not need to be linked with any of the opencl common files
createcsr_LDADD = include/common_util.o
createcsr_LINK = $(CCLD) -lm -o $@

all_local += csr-all-local
exec_local += csr-exec-local

csr-all-local:
	cp $(top_srcdir)/sparse-linear-algebra/SPMV/src/spmv_csr_kernel.cl .

csr-exec-local:
	cp $(top_srcdir)/sparse-linear-algebra/SPMV/src/spmv_csr_kernel.cl ${DESTDIR}${bindir}
