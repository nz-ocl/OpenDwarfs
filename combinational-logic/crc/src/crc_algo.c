#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "../../../include/rdtsc.h"
#include "../../../include/common_ocl.h"
#include "../inc/crc_formats.h"

#define DATA_SIZE 100000000
#define MIN(a,b) a < b ? a : b
unsigned char verbosity=0;
int platform_id=PLATFORM_ID, n_device=DEVICE_ID;

#ifdef USE_AFPGA
	const char *KernelSourceFile = "crc_kernel.aocx";
#else //CPU or GPU
	const char *KernelSourceFile = "crc_algo_kernel.cl";
#endif

cl_device_id device_id;
cl_context context;
cl_command_queue write_queue,kernel_queue,read_queue;
cl_program program;
cl_kernel kernel_compute;
cl_mem dev_table;


int64_t dataTransferTime = 0;
int64_t kernelExecutionTime = 0;

unsigned int page_size = DATA_SIZE;
unsigned char* h_tables;
unsigned char crc_polynomial=0x9B;

void usage()
{
	printf("crc [pd][hsivpw]\n");
	printf("Common arguments:\n");
	ocd_usage();
	printf("Program-specific arguments:\n");
	printf("\t-h | 'Print this help message'\n");
	printf("\t-v | 'Be Verbose'\n");
	printf("\t-s | 'Set random seed' [integer]\n");
	printf("\t-i | 'Input file name' [string]\n");
	printf("\t-a | 'Verify results on CPU'\n");
	printf("\t-p | 'CRC Polynomial' [integer]\n");
	printf("\t-n | 'Data size' [integer]\n");
	printf("\t-q | 'Kernel iterations' [integer]\n");
	printf("\t-w | 'Data block size' [integer]\n");

	printf("\nNOTE: Seperate common arguments and program specific arguments with the '--' delimeter\n");
	exit(0);
}

void printTimeDiff(struct timeval start, struct timeval end)
{
  printf("%ld microseconds\n", ((end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec)));
}

int64_t computeTimeDiff(struct timeval start, struct timeval end)
{
	int64_t diff = (end.tv_sec * 1000000 + end.tv_usec)
		  - (start.tv_sec * 1000000 + start.tv_usec);
	return diff;
}

unsigned char serialCRC(unsigned char* h_num, size_t size, unsigned char crc)
{
	unsigned int i;
	unsigned char num = h_num[0];
	for(i = 1; i < size + 1; i++)
	{
		unsigned char crcCalc = h_num[i];
		unsigned int k;
		if(i == size)
			crcCalc = 0;
		for(k = 0; k < 8; k++)
		{
			//If the k-th bit is 1
			if((num >> (7-k)) % 2 == 1)
			{
				num ^= crc >> (k + 1);
				crcCalc ^= crc << (7-k);
			}
		}
		num = crcCalc;
	}

	return num;
}

void computeTables(unsigned char* tables, int numTables, unsigned char crc)
{
	int level = 0;
	int i;
	for(i = 0; i < 256; i++)
	{
		unsigned char val = i;
		tables[i] = serialCRC(&val, 1, crc);
	}
	for(level = 1; level < numTables; level++)
	{
		for(i = 0; i < 256; i++)
		{
			unsigned char val = tables[(level-1)*256 + i];
			tables[level * 256 + i] = tables[(level-1)*256 + val];
		}
	}
}

int prepare_tables()
{
	cl_int err;
	int numTables;

	numTables = floor(log(page_size)/log(2)) + 1; //Generate Tables for the given size of page_size
	printf("num tables = %d\n", numTables);
	h_tables = char_new_array(256*numTables,"crc_algo.main() - Heap Overflow! Cannot allocate space for h_tables");

	dev_table = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(char)*256 * numTables, NULL, &err);
	CHKERR(err, "Failed to allocate device memory!");

	computeTables(h_tables, numTables, crc_polynomial);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(write_queue, dev_table, CL_TRUE, 0, sizeof(char)*256*numTables, h_tables, 0, NULL, &ocdTempEvent);
	clFinish(write_queue);
	CHKERR(err, "Failed to write to source array!");
	START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "CRC Look-Up Tables Copy", ocdTempTimer)
	END_TIMER(ocdTempTimer)

	return numTables;
}

unsigned char computeCRCDevice(unsigned char* h_num, int numTables, cl_mem d_input, cl_mem d_output,cl_event* write_page,cl_event* kernel_exec,cl_event* read_page)
{
	if(verbosity) printf("Running Kernel\n");
	size_t global_size,local_size;
	unsigned char* h_answer;

	// Write our data set into the input array in device memory
	int err = clEnqueueWriteBuffer(write_queue, d_input, CL_FALSE, 0, sizeof(char)*page_size, h_num, 0, NULL, write_page);
	CHKERR(err, "Failed to write to source array!");

	// Set the arguments to our compute kernel
	err = clSetKernelArg(kernel_compute, 0, sizeof(cl_mem), &d_input);
	err |= clSetKernelArg(kernel_compute, 1, sizeof(cl_mem), &dev_table);
	err |= clSetKernelArg(kernel_compute, 2, sizeof(unsigned int), &numTables);
	err |= clSetKernelArg(kernel_compute, 3, sizeof(cl_mem), &d_output);
	err |= clSetKernelArg(kernel_compute, 4, sizeof(unsigned int), &page_size);
	CHKERR(err, "Failed to set compute kernel arguments!");

	// Get the maximum work group size for executing the kernel on the device
	err = clGetKernelWorkGroupInfo(kernel_compute, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), (void *) &local_size, NULL);
	CHKERR(err, "Failed to retrieve kernel_compute work group info!");
	
	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	global_size = page_size + local_size - page_size%local_size;

	err = clEnqueueNDRangeKernel(kernel_queue, kernel_compute, 1, NULL, &global_size, &local_size, 1, write_page, kernel_exec);
	CHKERR(err, "Failed to execute compute kernel!");

	h_answer = char_new_array(page_size,"crc_algo.computeCRCGPU() - Heap Overflow! Cannot allocate space for h_answer");
	
	// Read back the results from the device to verify the output
	err = clEnqueueReadBuffer(read_queue, d_output, CL_FALSE, 0, sizeof(char)*page_size, h_answer, 1, kernel_exec, read_page);
	CHKERR(err, "Failed to read output array!");

	unsigned char answer = 0;
	
	int i;
	for(i = 0; i < page_size; i++)
		answer ^= h_answer[i];

	return answer;
}

void setup_device()
{
	cl_int err,dev_type;
	struct timeval compilation_st, compilation_et;
	char *kernelSource;
	size_t kernelLength;

	#ifdef USEGPU
		 dev_type = CL_DEVICE_TYPE_GPU;
	#elif defined(USE_AFPGA)
		 dev_type = CL_DEVICE_TYPE_ACCELERATOR;
	#else
		dev_type = CL_DEVICE_TYPE_CPU;
	#endif

	if(verbosity) printf("Getting Device\n");
	device_id = GetDevice(platform_id, n_device,dev_type);

	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	CHKERR(err, "Failed to create a compute context!");

	/* Create command queues, one for each stage in the write-execute-read pipeline */
	write_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	CHKERR(err, "Failed to create a command queue!");
	kernel_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	CHKERR(err, "Failed to create a command queue!");
	read_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	CHKERR(err, "Failed to create a command queue!");
	
	gettimeofday(&compilation_st, NULL);
	program = ocdBuildProgramFromFile(context,device_id,KernelSourceFile);

	// Create the compute kernel in the program we wish to run
	kernel_compute = clCreateKernel(program, "compute", &err);
	CHKERR(err, "Failed to create a compute kernel!");
	gettimeofday(&compilation_et, NULL);

	printf("Kernel Compilation Time: ");
	printTimeDiff(compilation_st, compilation_et);
}

int main(int argc, char** argv)
{
	cl_int err;
	size_t maxSize=DATA_SIZE;
	FILE* fp=NULL;
	unsigned char **h_num,ocl_remainder,serial_remainder;
	unsigned int run_serial=0,seed=time(NULL),i,num_pages=1;
	char* file=NULL;
	int c,numTables;

	ocd_requirements req;
	ocd_parse(&argc, &argv);
	ocd_check_requirements(NULL);
	
	while((c = getopt (argc, argv, "avn:s:i:p:w:h")) != -1)
	{
		switch(c)
		{
			case 'h':
				usage();
				exit(0);
				break;
			case 'p':
				crc_polynomial = atoi(optarg);
				break;
			case 'v':
				verbosity=1;
				break;
			case 'a':
				run_serial = 1;
				break;
			case 'i':
				if(optarg != NULL)
					file = optarg;
				else
					file = argv[optind];
				printf("Reading Input from '%s'\n",file);
				break;
			case 's':
				seed = atoi(optarg);
				break;
			case 'n':
				page_size = atoi(optarg);
				check(page_size != 0,"page_size must be greater than 0!");
				break;
			case 'w':
				maxSize = atoi(optarg);
				break;
			default:
				fprintf(stderr, "Invalid argument: '%s'\n\n",optarg);
				usage();
		}	
	}

	if(!file)
	{
		h_num = rand_crc(num_pages,page_size,seed);
	}
	else
	{
		h_num = read_crc(&num_pages,&page_size,file);
	}

	cl_mem dev_output[num_pages],dev_input[num_pages];
	cl_event write_page[num_pages],kernel_exec[num_pages],read_page[num_pages];

	ocd_options opts = ocd_get_options();
	platform_id = opts.platform_id;
	n_device = opts.device_id;

	if(verbosity) printf("Common Arguments: p=%d d=%d\n", platform_id, n_device);
	if(verbosity) printf("Program Arguments: p=%d v=%d i=%s s=%d n=%lu w=%d\n", crc_polynomial, run_serial, file, seed, page_size, (int)maxSize);

	if(verbosity) printf("Setting up device...\n");
	setup_device();
	numTables = prepare_tables();

	for(i=0; i<num_pages; i++)
	{
		dev_input[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char)*page_size, NULL, &err);
		CHKERR(err, "Failed to allocate device memory!");
		dev_output[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char)*page_size, NULL, &err);
		CHKERR(err, "Failed to allocate device memory!");
	}

	#ifdef ENABLE_TIMER
		TIMER_INIT
	#endif
	for(i=0; i<num_pages; i++)
	{
		ocl_remainder = computeCRCDevice(h_num[i],numTables,dev_input[i],dev_output[i],&write_page[i],&kernel_exec[i],&read_page[i]);
		if(verbosity) printf("Parallel CRC: '%X'\n", ocl_remainder);
	}
	clFinish(write_queue);
	clFinish(kernel_queue);
	clFinish(read_queue);

	#ifdef ENABLE_TIMER
		TIMER_STOP
	#endif

	for(i=0; i<num_pages; i++)
	{
		START_TIMER(write_page[i], OCD_TIMER_H2D, "CRC Data Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		START_TIMER(kernel_exec[i], OCD_TIMER_KERNEL, "CRC Kernel", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		START_TIMER(read_page[i], OCD_TIMER_D2H, "CRC Data Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)
	}

	#ifdef ENABLE_TIMER
		TIMER_PRINT
	#endif

	if(run_serial) // verify that we have the correct answer with regular C
	{
		printf("Computing Serial CRC\n");
		for(i=0; i<num_pages; i++)
			serial_remainder = serialCRC(h_num[i], page_size, crc_polynomial);
		if(verbosity) printf("Serial Computation: '%X'\n", serial_remainder);
	}	

	free(h_num);
	free(h_tables);
	#ifdef ENABLE_TIMER
		TIMER_DEST
	#endif
	for(i=0; i<num_pages; i++)
	{
		clReleaseMemObject(dev_input[i]);
		clReleaseMemObject(dev_output[i]);
		clReleaseEvent(write_page[i]);
		clReleaseEvent(kernel_exec[i]);
		clReleaseEvent(read_page[i]);
	}
	clReleaseKernel(kernel_compute);
	clReleaseCommandQueue(write_queue);
	clReleaseCommandQueue(kernel_queue);
	clReleaseCommandQueue(read_queue);
	clReleaseContext(context);

	return 0;
}
