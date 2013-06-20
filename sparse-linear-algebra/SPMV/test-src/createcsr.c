#include "../inc/sparse_formats.h"
#include "../inc/sparse_formats.c"

#include <stdio.h>
#include <stdlib.h>
#include<sys/types.h>
#include<unistd.h>
#include <getopt.h>

#define CSR_NAME_MAX_LENGTH 256

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"print", 0, NULL, 'p'},
      {"density",1,NULL, 'd'},
      {"size",1,NULL, 'n'},
      {"stddev",1,NULL, 's'},
      {"csr-file",1,NULL,'f'},
      {"no-rand",0,NULL,'R'},
      {"no-save",0,NULL,'S'},
      {0,0,0,0}
};

int main(int argc, char** argv)
{
  unsigned int N = 512;
  unsigned int density = 10000;
  unsigned long seed=10000;
  double normal_stddev = .35;
  char *file_path=NULL,do_print=0,do_rand=1,do_save=1,free_file=0;
  time_t t;
  struct tm tm;
  int opt,option_index=0,normal_stddev_rounded;

  const char* usage = "Usage: %s [-n <size>] [-d <d_ppm>] [-s <n_stddev>] [-p] [-f <file_path>] [-R] [-S]\n\n \
		  	-n: Set length and width of matrix to <size> - Default is 512\n \
      		-d: Set density of matrix (fraction of Non-Zero Elements) to <d_ppm> / 1,000,000 - Default is 10,000 (10%)\n \
		    -s: Set standard deviation of NNZ/Row to n_stddev * (Average NNZ/ROW) - Default is .35\n \
      		-f: Save CSR Matrix to file <file_path> rather than the default csrmatrix_N<size>D<d_ppm>_<year>-<month>-<day>-<hour>-<min>\n \
      		-p: Print matrix to stdout in standard (2-D Array) format\n \
		    -R: Do NOT seed random number generator in order to produce repeatable results\n \
		    -S: Do NOT save the matrix to a file\n\n";

  while ((opt = getopt_long(argc, argv, "n:d:s:f:pRS", long_options, &option_index)) != -1 )
  {
  	switch(opt)
	{
  		case 'n':
			if(optarg != NULL)
				N = atoi(optarg);
			else
				N = atoi(argv[optind]);
			break;
  		case 'd':
			if(optarg != NULL)
				density = atol(optarg);
			else
				density = atol(argv[optind]);
			break;
  		case 's':
  			if(optarg != NULL)
  				normal_stddev = atof(optarg);
  			else
  				normal_stddev = atof(argv[optind]);
  			break;
		case 'f':
			if(optarg != NULL)
				file_path = optarg;
			else
				file_path = argv[optind];
			break;
		case 'p':
			do_print = 1;
			break;
		case 'R':
			do_rand = 0;
			break;
		case 'S':
			do_save = 0;
			break;
		default:
			fprintf(stderr, usage,argv[0]);
			exit(EXIT_FAILURE);
	  }
  }

  if(do_rand) seed = (unsigned long) getpid();

  printf("Generating Matrix...\n\n");

  csr_matrix csr = rand_csr(N,density,normal_stddev,seed,stdout);

  if(do_print) print_csr_std(&csr,stdout);
  else print_csr_metadata(&csr,stdout);

  if(do_save)
  {
	if(!file_path)
	{
		file_path = malloc(sizeof(char)*CSR_NAME_MAX_LENGTH);
		t = time(NULL);
		tm = *localtime(&t);
		normal_stddev_rounded = (int) round(normal_stddev * 100);
		snprintf(file_path,(sizeof(char)*CSR_NAME_MAX_LENGTH),"../test/sparse-linear-algebra/SPMV/csrmatrix_N%lu_D%lu_S%d_%d-%d-%d-%d-%d",N,density,normal_stddev_rounded,(tm.tm_year+1900),tm.tm_mon+1,tm.tm_mday,tm.tm_hour,tm.tm_min);
		free_file=1;
	}
	printf("Saving Matrix to File '%s'...\n\n",file_path);
	write_csr(&csr,file_path);
  }
  if(free_file) free(file_path);
  free_csr(&csr);

  return 0;
}
