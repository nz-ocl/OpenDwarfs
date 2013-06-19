#include "sparse_formats.h"
#include "ziggurat.h"
#include "ziggurat.c"
#include<stdlib.h>

void chck(int b, const char* msg)
{
	if(!b)
	{
		fprintf(stderr,"error: %s\n\n",msg);
		exit(-1);
	}
}

unsigned int * int_new_array(const size_t N) {
    //dispatch on location
    return (unsigned int*) malloc(N * sizeof(unsigned int));
}

unsigned long * long_new_array(const size_t N) {
	//dispatch on location
	return (unsigned long*) malloc(N * sizeof(unsigned long));
}

float * float_new_array(const size_t N) {
    //dispatch on location
    return (float*) malloc(N * sizeof(float));
}

int unsigned_int_comparator(const void* v1, const void* v2)
{
	const unsigned int int1 = *((unsigned int*) v1);
	const unsigned int int2 = *((unsigned int*) v2);

	if(int1 < int2)
		return -1;
	else if(int1 > int2)
		return +1;
	else
		return 0;
}

void write_csr(const csr_matrix* csr,const char* file_path)
{
	FILE* fp;
	int i;
	fp = fopen(file_path,"w");
	chck(fp != NULL,"sparse_formats.write_csr() - Cannot Open File");

	fprintf(fp,"%u\n%f\n%u\n%u\n%u\n",csr->index_type,csr->value_type,csr->num_rows,csr->num_cols,csr->num_nonzeros);

	for(i=0; i<=csr->num_rows; i++)
	  fprintf(fp,"%u ",csr->Ap[i]);
	fprintf(fp,"\n");

	for(i=0; i<csr->num_nonzeros; i++)
	  fprintf(fp,"%u ",csr->Aj[i]);
	fprintf(fp,"\n");

	for(i=0; i<csr->num_nonzeros; i++)
	  fprintf(fp,"%f ",csr->Ax[i]);
	fprintf(fp,"\n");

	fclose(fp);
}

void read_csr(csr_matrix* csr,const char* file_path)
{
	FILE* fp;
	int i,read_count;

	fp = fopen(file_path,"r");
	chck(fp != NULL,"sparse_formats.read_csr() - Cannot Open Input File");

	read_count = fscanf(fp,"%u\n%f\n%u\n%u\n%u\n",&(csr->index_type),&(csr->value_type),&(csr->num_rows),&(csr->num_cols),&(csr->num_nonzeros));
	chck(read_count == 5,"sparse_formats.read_csr() - Input File Corrupted! Read count for header info differs from 5");

	read_count = 0;
	csr->Ap = int_new_array(csr->num_rows+1);
	chck((csr->Ap) != NULL,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Ap");
	for(i=0; i<=csr->num_rows; i++)
	  read_count += fscanf(fp,"%u ",csr->Ap+i);
	chck(read_count == (csr->num_rows+1),"sparse_formats.read_csr() - Input File Corrupted! Read count for Ap differs from csr->num_rows+1");

	read_count = 0;
	csr->Aj = int_new_array(csr->num_nonzeros);
	chck(csr->Aj != NULL,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Aj");
	for(i=0; i<csr->num_nonzeros; i++)
	  read_count += fscanf(fp,"%u ",csr->Aj+i);
	chck(read_count == (csr->num_nonzeros),"sparse_formats.read_csr() - Input File Corrupted! Read count for Aj differs from csr->num_nonzeros");

	read_count = 0;
	csr->Ax = float_new_array(csr->num_nonzeros);
	chck(csr->Ax != NULL,"sparse_formats.read_csr() - Heap Overflow! Cannot allocate space for csr.Ax");
	for(i=0; i<csr->num_nonzeros; i++)
	  read_count += fscanf(fp,"%f ",csr->Ax+i);
	chck(read_count == (csr->num_nonzeros),"sparse_formats.read_csr() - Input File Corrupted! Read count for Ax differs from csr->num_nonzeros");

	fclose(fp);
}

void print_timestamp(FILE* stream)
{
  time_t rawtime;
  struct tm* timeinfo;

  time(&rawtime);
  timeinfo = localtime(&rawtime);
  fprintf(stream,"Current time: %s",asctime(timeinfo));
}

unsigned long gen_rand(const long LB, const long HB)
{
	int range = HB - LB + 1;
	chck((HB >= 0 && LB >= 0 && range > 0),"sparse_formats.gen_rand() - Invalid Bound(s). Exiting...");
    return (rand() % range) + LB;
}

void print_csr_metadata(const csr_matrix* csr, FILE* stream) {
	fprintf(stream,"\nCSR Matrix Metadata:\n\nNRows=%lu\tNCols=%lu\tNNZ=%lu\tDensity=%lu ppm = %g%%\tAverage NZ/Row=%g\tStdDev NZ/Row=%g\n\n",csr->num_rows,csr->num_cols,csr->num_nonzeros,csr->density_ppm,csr->density_perc,csr->nz_per_row,csr->stddev);
}

void print_csr_std(const csr_matrix* csr,FILE* stream)
{
	int ind,ind2,nz_count=0,row_count=0,next_nz_row;
	float val,density;
	density = ((float)(csr->num_nonzeros))/(((float)(csr->num_rows))*((float)(csr->num_cols)));

	print_csr_metadata(csr,stream);

	while(csr->Ap[row_count+1] == nz_count)
		row_count++;

	for(ind=0; ind<csr->num_rows; ind++)
	{
		fprintf(stream,"[");
		for(ind2=0; ind2<csr->num_cols; ind2++)
		{
			if(ind == row_count && ind2 == csr->Aj[nz_count])
			{
				val = csr->Ax[nz_count++];
				while(csr->Ap[row_count+1] == nz_count)
					row_count++;
			}
			else
				val = 0.0;
			fprintf(stream,"%6.2f",val);
		}
		fprintf(stream,"]\n");
	}
	fprintf(stream,"\n");
}

csr_matrix rand_csr(const unsigned int N,const unsigned int density, const double normal_stddev,unsigned long seed,FILE* log)
{
	unsigned int i,j,nnz_ith_row,nnz,update_interval,rand_col;
	double nnz_ith_row_double,nz_error;
	int kn[128];
	float fn[128],wn[128];
	char* used_cols;
	csr_matrix csr;

	csr.num_rows = N;
	csr.num_cols = N;
	csr.density_perc = (((double)(density))/10000.0);
	csr.nz_per_row = (((double)(N*density))/1000000.0);
	csr.num_nonzeros = round(csr.nz_per_row*N);
	csr.stddev = normal_stddev * csr.nz_per_row; //scale normalized standard deviation by average NZ/row

	fprintf(log,"Average NZ/Row: %-8.3f\n",csr.nz_per_row);
	fprintf(log,"Standard Deviation: %-8.3f\n",csr.stddev);
	fprintf(log,"Target Density: %u ppm = %g%%\n",density,csr.density_perc);
	fprintf(log,"Approximate NUM_nonzeros: %d\n",csr.num_nonzeros);

	csr.Ap = int_new_array(csr.num_rows+1);
	chck(csr.Ap != NULL,"rand_square_csr2() - Heap Overflow! Cannot Allocate Space for csr.Ap");
	csr.Aj = int_new_array(csr.num_nonzeros);
	chck(csr.Aj != NULL,"rand_square_csr2() - Heap Overflow! Cannot Allocate Space for csr.Aj");

	csr.Ap[0] = 0;
	nnz = 0;
	used_cols = malloc(csr.num_cols*sizeof(char));
	chck(used_cols != NULL,"rand_square_csr2() - Heap Overflow! Cannot allocate space for used_cols");

	r4_nor_setup(kn,fn,wn);

	update_interval = round(csr.num_rows / 10.0);
	if(!update_interval) update_interval = csr.num_rows;

	for(i=0; i<csr.num_rows; i++)
	{
		if(i % update_interval == 0) fprintf(log,"\t%d of %d (%5.1f%%) Rows Generated. Continuing...\n",i,csr.num_rows,((double)(i))/csr.num_rows*100);

		nnz_ith_row_double = r4_nor(&seed,kn,fn,wn); //random, normally-distributed value for # of nz elements in ith row, NORMALIZED
		nnz_ith_row_double *= csr.stddev; //scale by standard deviation
		nnz_ith_row_double += csr.nz_per_row; //add average nz/row
		if(nnz_ith_row_double < 0)
			nnz_ith_row = 0;
		else if(nnz_ith_row_double > csr.num_cols)
			nnz_ith_row = csr.num_cols;
		else
			nnz_ith_row = (unsigned int) round(nnz_ith_row_double);

		csr.Ap[i+1] = csr.Ap[i] + nnz_ith_row;
		if(csr.Ap[i+1] > csr.num_nonzeros)
			csr.Aj = realloc(csr.Aj,sizeof(unsigned int)*csr.Ap[i+1]);

		for(j=0; j<csr.num_cols; j++)
			used_cols[j] = 0;

		for(j=0; j<nnz_ith_row; j++)
		{
			rand_col = gen_rand(0,csr.num_cols - 1);
			if(used_cols[rand_col])
			{
				j--;
			}
			else
			{
				csr.Aj[csr.Ap[i]+j] = rand_col;
				used_cols[rand_col] = 1;
			}
		}
		qsort((&(csr.Aj[csr.Ap[i]])),nnz_ith_row,sizeof(unsigned int),unsigned_int_comparator);
	}

	nz_error = ((double)abs((signed int)(csr.num_nonzeros - csr.Ap[csr.num_rows]))) / ((double)csr.num_nonzeros);
	if(nz_error >= .05)
		fprintf(stderr,"WARNING: Actual NNZ differs from Theoretical NNZ by %5.2f%%!\n",nz_error*100);
	csr.num_nonzeros = csr.Ap[csr.num_rows];
	fprintf(log,"Actual NUM_nonzeros: %d\n",csr.num_nonzeros);
	csr.density_perc = (double) (((double)(csr.num_nonzeros*100))/((double)csr.num_cols))/((double)csr.num_rows);
	csr.density_ppm = (unsigned int)round(csr.density_perc * 10000.0);
	fprintf(log,"Actual Density: %u ppm = %g%%\n",csr.density_ppm,csr.density_perc);

	free(used_cols);
	csr.Ax = float_new_array(csr.num_nonzeros);
	chck(csr.Ax != NULL,"rand_square_csr2() - Heap Overflow! Cannot Allocate Space for csr.Ax");
	for(i=0; i<csr.num_nonzeros; i++)
	{
		csr.Ax[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
		while(csr.Ax[i] == 0.0)
			csr.Ax[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0));
	}

	return csr;
}

void free_csr(csr_matrix* csr)
{
	free(csr->Ap);
	free(csr->Aj);
	free(csr->Ax);
}
