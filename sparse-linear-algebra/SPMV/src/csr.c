#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <assert.h>
#include <errno.h>

#include "../../../include/rdtsc.h"
#include "../../../include/common_ocl.h"
#include "../../../include/common_util.h"
#include "../inc/common.h"
#include "../inc/sparse_formats.h"

#define START_GTOD_TIMER { \
		gettimeofday(tv,NULL); \
    	start_time = 1000 * (tv->tv_sec*1000000L + tv->tv_usec); }

#define END_GTOD_TIMER { \
		gettimeofday(tv,NULL); \
		end_time = 1000 * (tv->tv_sec*1000000L + tv->tv_usec); }

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"cpu", 0, NULL, 'c'},
      {"device", 1, NULL, 'd'},
      {"verbose", 0, NULL, 'v'},
      {"input_file",1,NULL,'i'},
      {"print",0,NULL,'p'},
      {"affirm",0,NULL,'a'},
      {"repeat",1,NULL,'r'},
      {"kernel_file",1,NULL,'k'},
      {"wg_size",1,NULL,'w'},
      {0,0,0,0}
};

int platform_id=PLATFORM_ID, n_device=DEVICE_ID;

/**
 * Compares N float values and prints error msg if any corresponding entries differ by greater than .001
 */
void float_array_comp(const float* a, const float* b, const unsigned int N)
{
	unsigned int j;
	double diff,perc,avg;
	for (j = 0; j < N; j++)
	{
		diff = fabsf(a[j] - b[j]);
		if(diff > .001)
		{
			avg = (a[j] + b[j])/2.0;
			perc = diff/fabsf(avg) * 100;
			fprintf(stderr,"Possible error, difference of %.3f (%.1f%% error) [a=%.3f, b=%.3f, avg=%.3f] at row %d \n", diff,perc,a[j],b[j],avg,j);
		}
	}
}

/**
 * Sparse Matrix-Vector Multiply
 *
 * Multiplies csr matrix by vector x, adds vector y, and stores output in vector out
 */
void spmv_csr_cpu(const csr_matrix* csr,const float* x,const float* y,float* out)
{
   unsigned int row,row_start,row_end,jj;
   float sum = 0;
   for(row=0; row < csr->num_rows; row++)
   {
		sum = y[row];
		row_start = csr->Ap[row];
		row_end   = csr->Ap[row+1];

		for (jj = row_start; jj < row_end; jj++){
			sum += csr->Ax[jj] * x[csr->Aj[jj]];
		}
		out[row] = sum;
	}
}

size_t* check_wg_sizes(size_t* wg_sizes,unsigned int* num_wg_sizes,const size_t max_wg_size,const size_t global_size)
{
	unsigned int ii,num_wg;
	size_t min_wg_size=0;
	if(wg_sizes) //if wg_size was specified on command line
	{
		for(ii=0; ii<*num_wg_sizes; ii++)
		{
			printf("Max: %zd\tMin: %zd\tii: %u\twg_sizes[ii]: %zd\n",max_wg_size,min_wg_size,ii,wg_sizes[ii]);
			check(wg_sizes[ii] > 0 && wg_sizes[ii] <= max_wg_size,"csr.check_wg_sizes(): Illegal wg_size!");
		}
	}
	else
	{
		(*num_wg_sizes)=1;
		wg_sizes = malloc(sizeof(size_t)*(*num_wg_sizes));
		check(wg_sizes != NULL,"csr.main() - Heap Overflow! Cannot allocate space for wg_sizes");
		wg_sizes[0] = max_wg_size;
		num_wg = global_size / wg_sizes[0];
		if(global_size % wg_sizes[0] != 0) //if wg_size is not a factor of global_size
		{							//use min num_wg such that wg_size < global_size
			num_wg++;
			wg_sizes[0] = global_size / (num_wg);
		}
	}
	return wg_sizes;
}

void csrCreateBuffer(const cl_context* p_context, cl_mem* ptr, const size_t num_bytes, const cl_mem_flags flags, const char* buff_name, int be_verbose)
{
	cl_int err;
	char err_msg[128];
	if(be_verbose) printf("Allocating %zu bytes for %s...\n",num_bytes,buff_name);
	*ptr = clCreateBuffer(*p_context, flags,num_bytes, NULL, &err);
	snprintf(err_msg,88,"Failed to allocate device memory for %s!",buff_name);
	CHKERR(err, err_msg);
}

int main(int argc, char** argv)
{
	cl_int err;
	int num_wg,be_verbose = 0,do_print=0,do_affirm=0,do_mem_align=0,opt, option_index=0;
    unsigned long density_ppm = 500000;
    unsigned int num_execs=1,i,ii,iii,j,num_wg_sizes=0,num_kernels=0;
    unsigned long start_time, end_time;
	struct timeval *tv;
    char* file_path = NULL,*optptr;
    void* tmp;

    const char* usage = "Usage: %s -i <file_path> [-v] [-c] [-p] [-a] [-r <num_execs>] [-k <kernel_file-1>][-k <kernel_file-2>]...[-k <kernel_file-n>] [-w <wg_size-1>][-w <wg_size-2>]...[-w <wg_size-m>]\n\n \
    		-i: Read CSR Matrix from file <file_path>\n \
    		-k: Test SPMV 'n' times, once with each kernel_file-'1..n' - Default is 1 kernel named './spmv_csr_kernel.xxx' where xxx is 'aocx' if USE_AFPGA is defined, 'cl' otherwise.\n \
    		-v: Be Verbose \n \
    		-c: use CPU\n \
    		-p: Print matrices to stdout in standard (2-D Array) format - Warning: lots of output\n \
    		-a: Affirm results with serial C code on CPU\n \
    		-r: Execute program with same data exactly <num_execs> times to increase sample size - Default is 1\n \
    		-w: Loop through each kernel execution 'm' times, once with each wg_size-'1..m' - Default is 1 iteration with wg_size set to the maximum possible (limited either by the device or the size of the input)\n\n";

    size_t global_size;
    size_t* wg_sizes = NULL;
    size_t max_wg_size,kernelLength,items_read;

    cl_device_id device_id;
    cl_int dev_type;
    cl_context context;
    cl_command_queue commands;
    cl_program program;
    cl_kernel kernel;

    cl_mem csr_ap,csr_aj,csr_ax,x_loc,y_loc;

    FILE *kernel_fp;
    char *kernelSource,**kernel_files=NULL,*kernel_file_mode;

    ocd_parse(&argc, &argv);
	ocd_check_requirements(NULL);
    ocd_options opts = ocd_get_options();
    platform_id = opts.platform_id;
    n_device = opts.device_id;

	#ifdef USEGPU
    	 dev_type = CL_DEVICE_TYPE_GPU;
	#elif defined(USE_AFPGA)
    	 dev_type = CL_DEVICE_TYPE_ACCELERATOR;
	#else
    	dev_type = CL_DEVICE_TYPE_CPU;
	#endif

    while ((opt = getopt_long(argc, argv, "::vcmw:k:i:par:::", long_options, &option_index)) != -1 )
    {
    	switch(opt)
		{
    		case 'v':
    			printf("verify\n");
    			be_verbose = 1;
    			break;
			case 'c':
				printf("using cpu\n");
				dev_type = CL_DEVICE_TYPE_CPU;
				break;
			case 'i':
				if(optarg != NULL)
					file_path = optarg;
				else
					file_path = argv[optind];
				printf("Reading Input from '%s'\n",file_path);
				break;
			case 'p':
				do_print = 1;
				break;
			case 'a':
				do_affirm = 1;
				break;
			case 'r':
				if(optarg != NULL)
					num_execs = atoi(optarg);
				else
					num_execs = atoi(argv[optind]);
				printf("Executing %d times\n",num_execs);
				break;
			case 'k':
				if(optarg != NULL)
					optptr = optarg;
				else
					optptr = argv[optind];
				num_kernels++;
				tmp = realloc(kernel_files,sizeof(char*)*num_kernels);
				check(tmp != NULL,"csr.main() - Heap Overflow! Cannot allocate space for kernel_files");
				kernel_files = tmp;
				kernel_files[num_kernels-1] = optptr;
				printf("Testing with Kernel File: '%s'\n",kernel_files[num_kernels-1]);
				break;
			case 'w':
				if(optarg != NULL)
					optptr = optarg;
				else
					optptr = argv[optind];
				num_wg_sizes++;
				tmp = realloc(wg_sizes,sizeof(size_t)*num_wg_sizes);
				check(tmp != NULL,"csr.main() - Heap Overflow! Cannot allocate space for wg_sizes");
				wg_sizes = tmp;
				wg_sizes[num_wg_sizes-1] = atoi(optptr);
				break;
			default:
				fprintf(stderr, usage,argv[0]);
				exit(EXIT_FAILURE);
		  }
    }

    if(!file_path)
	{
    	fprintf(stderr,"-i Option must be supplied\n\n");
		fprintf(stderr, usage,argv[0]);
		exit(EXIT_FAILURE);
	}

    csr_matrix csr;
    read_csr(&csr,file_path);

    if(do_print) print_csr_std(&csr,stdout);
    else if(be_verbose) print_csr_metadata(&csr,stdout);

    //The other arrays
    float *x_host, *y_host, *device_out, *host_out;
	x_host = float_new_array(csr.num_cols,"csr.main() - Heap Overflow! Cannot Allocate Space for x_host");
	y_host = float_new_array(csr.num_rows,"csr.main() - Heap Overflow! Cannot Allocate Space for y_host");
	device_out = float_new_array(csr.num_rows,"csr.main() - Heap Overflow! Cannot Allocate Space for device_out");

	if(do_affirm)
	{
		host_out = malloc(sizeof(float)*csr.num_rows);
		check(host_out != NULL,"csr.main() - Heap Overflow! Cannot Allocate Space for 'host_out'");
	}

    for(ii = 0; ii < csr.num_cols; ii++)
    {
        x_host[ii] = rand() / (RAND_MAX + 1.0);
        if(do_print) printf("x[%d] = %6.2f\n",ii,x_host[ii]);
    }
    for(ii = 0; ii < csr.num_rows; ii++)
    {
        y_host[ii] = rand() / (RAND_MAX + 2.0);
        if(do_print) printf("y[%d] = %6.2f\n",ii,y_host[ii]);
    }

    if(be_verbose) printf("Input Generated.\n");

	#ifdef ENABLE_TIMER
		tv = malloc(sizeof(struct timeval));
		check(tv != NULL,"csr.main() - Heap Overflow! Cannot allocate space for tv");
	#endif

    /* Retrieve an OpenCL platform */
    device_id = GetDevice(platform_id, n_device,dev_type);

    if(be_verbose) ocd_print_device_info(device_id);



    if(!kernel_files) //use default if no kernel files were given on commandline
    {
		num_kernels = 1;
		kernel_files = malloc(sizeof(char*)*num_kernels);
		#ifdef USE_AFPGA
				kernel_files[0] = "spmv_csr_kernel.aocx";
		#else //CPU or GPU
			kernel_files[0] = "spmv_csr_kernel.cl";
		#endif
    }

	#ifdef USE_AFPGA
		kernel_file_mode = "rb";
	#else //CPU or GPU
		kernel_file_mode = "r";
	#endif

	for(iii=0; iii<num_kernels; iii++)
	{
	 /* Create a compute context */
		context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
		CHKERR(err, "Failed to create a compute context!");

		/* Create a command queue */
		commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
		CHKERR(err, "Failed to create a command queue!");

		csrCreateBuffer(&context,&csr_ap,sizeof(int)*(csr.num_rows+1),CL_MEM_READ_ONLY,"csr_ap",be_verbose);
		csrCreateBuffer(&context,&x_loc,sizeof(float)*csr.num_cols,CL_MEM_READ_ONLY,"x_loc",be_verbose);
		csrCreateBuffer(&context,&y_loc,sizeof(float)*csr.num_rows,CL_MEM_READ_WRITE,"y_loc",be_verbose);
		csrCreateBuffer(&context,&csr_aj,sizeof(int)*csr.num_nonzeros,CL_MEM_READ_ONLY,"csr_aj",be_verbose);
		csrCreateBuffer(&context,&csr_ax,sizeof(float)*csr.num_nonzeros,CL_MEM_READ_ONLY,"csr_ax",be_verbose);

		/* Load kernel source */
		printf("Kernel #%d: '%s'\n\n",iii+1,kernel_files[iii]);
		kernel_fp = fopen(kernel_files[iii], kernel_file_mode);
		check(kernel_fp != NULL,"Cannot open kernel file");
		fseek(kernel_fp, 0, SEEK_END);
		kernelLength = (size_t) ftell(kernel_fp);
		kernelSource = (char *) malloc(sizeof(char)*kernelLength);
		check(kernelSource != NULL,"csr.main() - Heap Overflow! Cannot allocate space for kernelSource.");
		rewind(kernel_fp);
		items_read = fread((void *) kernelSource, kernelLength, 1, kernel_fp);
		check(items_read == 1,"csr.main() - Error reading from kernelFile");
		fclose(kernel_fp);
		if(be_verbose) printf("kernel source loaded.\n");

		/* Create the compute program from the source buffer */
		#ifdef USE_AFPGA //use Altera FPGA
			program = clCreateProgramWithBinary(context,1,&device_id,&kernelLength,(const unsigned char**)&kernelSource,NULL,&err);
		#else //CPU or GPU
			program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, &kernelLength, &err);
		#endif
		CHKERR(err, "Failed to create a compute program!");

		free(kernelSource); /* Free kernel source */

		/* Build the program executable */
		#ifdef USE_AFPGA //use Altera FPGA
			err = clBuildProgram(program,1,&device_id,NULL,NULL,NULL);
		#else
			err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		#endif
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			char *buildLog;
			size_t logLen;
			err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLen);
			buildLog = (char *) malloc(sizeof(char)*logLen);
			check(buildLog != NULL,"csr.main() - Heap Overflow! Cannot allocate space for buildLog.");
			err = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logLen, (void *) buildLog, NULL);
			fprintf(stderr, "CL Error %d: Failed to build program! Log:\n%s", err, buildLog);
			free(buildLog);
			exit(1);
		}
		CHKERR(err, "Failed to build program!");

		/* Create the compute kernel in the program we wish to run */
		kernel = clCreateKernel(program, "csr", &err);
		CHKERR(err, "Failed to create a compute kernel!");
		if(be_verbose) printf("kernel created\n");

		/* Set the arguments to our compute kernel */
		err = clSetKernelArg(kernel, 0, sizeof(unsigned int), &csr.num_rows);
		err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &csr_ap);
		err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &csr_aj);
		err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &csr_ax);
		err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &x_loc);
		err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &y_loc);
		CHKERR(err, "Failed to set kernel arguments!");
		if(be_verbose) printf("set kernel arguments\n");

		/* Get the maximum work group size for executing the kernel on the device */
		err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *) &max_wg_size, NULL);
		if(be_verbose) printf("Kernel Max Work Group Size: %d\n",max_wg_size);
		CHKERR(err, "Failed to retrieve kernel work group info!");

		global_size = csr.num_rows;
		wg_sizes = check_wg_sizes(wg_sizes,&num_wg_sizes,max_wg_size,global_size);

		for(ii=0; ii<num_wg_sizes; ii++) //loop through all wg_sizes that need to be tested
		{
			num_wg = global_size / wg_sizes[ii];
			printf("WG Size #%d -- globalsize: %d - num_wg: %d - wg_size: %d\n",ii+1,global_size,num_wg,wg_sizes[ii]);

			for(i=0; i<num_execs; i++) //repeat Host-Device transfer, kernel execution, and device-host transfer num_execs times
			{						//to gather multiple samples of data
				if(be_verbose) printf("Beginning execution #%d of %d\n",i+1,num_execs);

				#ifdef ENABLE_TIMER
					TIMER_INIT
					START_GTOD_TIMER
				#endif

				/* Write our data set into the input array in device memory */
				err = clEnqueueWriteBuffer(commands, csr_ap, CL_TRUE, 0, sizeof(unsigned int)*csr.num_rows+4, csr.Ap, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				if(be_verbose) printf("Ap Buffer Written\n");
				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CSR Data Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to write to source array!");

				err = clEnqueueWriteBuffer(commands, csr_aj, CL_TRUE, 0, sizeof(unsigned int)*csr.num_nonzeros, csr.Aj, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				if(be_verbose) printf("Aj Buffer Written\n");
				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CSR Data Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to write to source array!");

				err = clEnqueueWriteBuffer(commands, csr_ax, CL_TRUE, 0, sizeof(float)*csr.num_nonzeros, csr.Ax, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				if(be_verbose) printf("Ax Buffer Written\n");
				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CSR Data Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to write to source array!");

				err = clEnqueueWriteBuffer(commands, x_loc, CL_TRUE, 0, sizeof(float)*csr.num_cols, x_host, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				if(be_verbose) printf("X_host Buffer Written\n");
				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CSR Data Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to write to source array!");

				err = clEnqueueWriteBuffer(commands, y_loc, CL_TRUE, 0, sizeof(float)*csr.num_rows, y_host, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				if(be_verbose) printf("Y_host Buffer Written\n");
				START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CSR Data Copy", ocdTempTimer)
				CHKERR(err, "Failed to write to source array!");
				END_TIMER(ocdTempTimer)

				#ifdef ENABLE_TIMER
					END_GTOD_TIMER
					if(be_verbose) printf("H2D GTOD:\t%llu\n",end_time - start_time);
					START_GTOD_TIMER
				#endif

				err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global_size, &wg_sizes[ii], 0, NULL, &ocdTempEvent);
				clFinish(commands);
				START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "CSR Kernel", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to execute kernel!");

				#ifdef ENABLE_TIMER
					END_GTOD_TIMER
					if(be_verbose) printf("Kernel GTOD:\t%llu\n",end_time - start_time);
					START_GTOD_TIMER
				#endif

				/* Read back the results from the device to verify the output */
				err = clEnqueueReadBuffer(commands, y_loc, CL_TRUE, 0, sizeof(float)*csr.num_rows, device_out, 0, NULL, &ocdTempEvent);
				clFinish(commands);
				START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "CSR Data Copy", ocdTempTimer)
				END_TIMER(ocdTempTimer)
				CHKERR(err, "Failed to read output array!");

				#ifdef ENABLE_TIMER
					END_GTOD_TIMER
					if(be_verbose) printf("D2H GTOD:\t%llu\n",end_time - start_time);
					TIMER_PRINT;
				#endif

				if(do_print)
				{
				   for (j = 0; j < csr.num_rows; j++)
					   printf("row: %d	output: %6.2f \n", j, device_out[j]);
				}

				if(do_affirm)
				{
				   if(be_verbose) printf("Validating results with serial C code on CPU...\n");
				   spmv_csr_cpu(&csr,x_host,y_host,host_out);
				   float_array_comp(host_out,device_out,csr.num_rows);
				}
			}
		}
		err = clReleaseMemObject(csr_ap);
		CHKERR(err,"Failed to release csr_ap!");
		if(be_verbose) printf("Released csr_ap\n");
		err = clReleaseMemObject(csr_aj);
		CHKERR(err,"Failed to release csr_aj!");
		if(be_verbose) printf("Released csr_aj\n");
		err = clReleaseMemObject(csr_ax);
		CHKERR(err,"Failed to release csr_ax!");
		if(be_verbose) printf("Released csr_ax\n");
		err = clReleaseMemObject(x_loc);
		CHKERR(err,"Failed to release x_loc!");
		if(be_verbose) printf("Released x_loc\n");
		err = clReleaseMemObject(y_loc);
		CHKERR(err,"Failed to release y_loc!");
		if(be_verbose) printf("Released y_loc\n");
		err = clReleaseProgram(program);
		CHKERR(err,"Failed to release program!");
		if(be_verbose) printf("Released program\n");
		err = clReleaseKernel(kernel);
		CHKERR(err,"Failed to release kernel!");
		if(be_verbose) printf("Released kernel\n");


	    clReleaseCommandQueue(commands);
	    CHKERR(err,"Failed to release commands!");
		if(be_verbose) printf("Released commands\n");

	    clReleaseContext(context);
	    CHKERR(err,"Failed to release context!");
		if(be_verbose) printf("Released context\n");

	}
	#ifdef ENABLE_TIMER
    	TIMER_FINISH;
	#endif

    /* Shutdown and cleanup */
    free(kernel_files);
    free(wg_sizes);
	free(tv);
    free(x_host);
    free(y_host);
    if(do_affirm) free(host_out);
    free_csr(&csr);
    free(device_out);

    return 0;
}

