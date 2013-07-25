#include "common_ocl.h"

ocd_options _settings = {0, 0, 0};
ocd_requirements _requirements = {0,0,0};
option* _options = NULL;

int _options_length = 0;
int _options_size = 0;

void _ocd_create_arguments()
{
	free(_options);
	_options = (option*)malloc(sizeof(option) * 5);
	option ops[3] = {{OTYPE_INT, 'p', (char*)"platform", (char*)"OpenCL Platform ID",
                     OFLAG_NONE, &_settings.platform_id, NULL, NULL, NULL, NULL},
		{OTYPE_INT, 'd', (char*)"device", (char*)"OpenCL Device ID",
                     OFLAG_NONE, &_settings.device_id, NULL, NULL, NULL, NULL},
		{OTYPE_END, '\0', (char*)"", NULL,
                     OFLAG_NONE, NULL, NULL, NULL, NULL, NULL}};
	
	_options[0] = ops[0];
	_options[1] = ops[1];
	_options[2] = ops[2];
	_options_length = 5;
	_options_size = 3;
}

ocd_options ocd_get_options()
{
	return _settings;
}

int ocd_parse(int* argc, char*** argv)
{
	if(!_options)
		_ocd_create_arguments();
	
	int largc = *argc;
	char** largv = *argv;

	//Determine if there are any arguments to parse.
	int i;
	int origargc = largc;
	for(i = 0; i < largc; i++)
	{
		if(strlen(largv[i]) == 2 && strncmp(largv[i], "--", 2) == 0)
		{
			largc = i;
			break;
		}
	}

	if(largc == origargc) // No double dash
	{
		//Assume we failed because they were actually
		//Program specific arguments.
		return 0;
	}

	if (optsgets(largc, largv, _options)) {

		fprintf(stderr, "optsgets() errored.\nUsage:");
		option* op;
		for (op = &_options[0]; op->type; ++op)
			if (!((op->type == OTYPE_NUL) &&
			  (op->flags & OFLAG_ARG)))
				fprintf(stderr, "%s\n", optsusage(op));
	}

	*argv += largc;
	*argc = *argc - largc;
	return largc;
}

cl_device_id _ocd_get_device(int platform, int device)
{
	cl_int err,dev_type;
	cl_uint nPlatforms = 1;

	#ifdef USE_AFPGA
	dev_type = CL_DEVICE_TYPE_ACCELERATOR;
	#elif defined(USEGPU)
	dev_type = CL_DEVICE_TYPE_GPU;
	#else
	dev_type = CL_DEVICE_TYPE_CPU;
	#endif

	err = clGetPlatformIDs(0, NULL, &nPlatforms);
	CHECK_ERROR(err);

	if (nPlatforms <= 0)
	{
		printf("No OpenCL platforms found. Exiting.\n");
		exit(0);
	}
	if (platform<0 || platform>=nPlatforms)  // platform ID out of range
	{
		printf("Platform index %d is out of range. \n", platform);
		exit(-4);
	}
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*nPlatforms);
	err = clGetPlatformIDs(nPlatforms, platforms, NULL);
	CHECK_ERROR(err);

	cl_uint nDevices = 1;
	char platformName[100];
	err = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, sizeof(platformName), platformName, NULL);
	CHECK_ERROR(err);
	printf("Platform Chosen : %s\n", platformName);
	// query devices
	err = clGetDeviceIDs(platforms[platform], dev_type, 0, NULL, &nDevices);
	CHECK_ERROR(err);
	if (nDevices <= 0)
	{
		printf("No OpenCL Device found. Exiting.\n");
		exit(0);
	}
	if (device<0 || device>=nDevices)  // platform ID out of range
	{
		printf("Device index %d is out of range. \n", device);
		exit(-4);
	}
	cl_device_id* devices = (cl_device_id *) malloc (sizeof(cl_device_id) * nDevices);
	err = clGetDeviceIDs(platforms[platform], dev_type, nDevices, devices, NULL);
	char DeviceName[100];
	err = clGetDeviceInfo(devices[device], CL_DEVICE_NAME, sizeof(DeviceName), DeviceName, NULL);
	CHECK_ERROR(err);
	printf("Device Chosen : %s\n", DeviceName);

	return devices[device];
}

int ocd_check_requirements(ocd_requirements* reqs)
{
	cl_int dev_type;

	if(reqs == NULL)
		return 1;

	int pass = 1;

	#ifdef USE_AFPGA
	dev_type = CL_DEVICE_TYPE_ACCELERATOR;
	#elif defined(USEGPU)
	dev_type = CL_DEVICE_TYPE_GPU;
	#else
	dev_type = CL_DEVICE_TYPE_CPU;
	#endif

	ocd_options opts = ocd_get_options();
//	cl_device_id d_id = _ocd_get_device(opts.platform_id, opts.device_id);
	cl_device_id d_id = GetDevice(opts.platform_id, opts.device_id,dev_type);

	cl_ulong local_mem;
	clGetDeviceInfo(d_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem, NULL);
	if(local_mem < reqs->local_mem_size)
		pass = 0;
	reqs->local_mem_size = local_mem;
	
	cl_ulong global_mem;
	clGetDeviceInfo(d_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, NULL);
	if(global_mem < reqs->global_mem_size)
		pass = 0;
	reqs->global_mem_size = global_mem;
	
	size_t workgroup_size;
	clGetDeviceInfo(d_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);
	if(workgroup_size < reqs->workgroup_size)
		pass = 0;
	reqs->workgroup_size = workgroup_size;

	return pass;
}
	

void _ocd_expand_list()
{
	option* new_options = (option*)malloc(sizeof(option) * (_options_length + 5));
	memcpy(new_options, _options, sizeof(option) * _options_length);
	free(_options);
	_options = new_options;
	_options_length += 5;
}

void _ocd_add_arg(option o, int size)
{
	if(_options_size >= _options_length)
		_ocd_expand_list();
	
	option end = _options[size-1];
	_options[size-1] = o;
	_options[size] = end;	
	
	_options_size++;
}

int ocd_register_arg(int type, char abbr, char* name, char* desc, void* value, optsverify verify, optssettor settor)
{
	option o = {type, abbr, name, desc, OFLAG_NONE, value, 0, verify, settor, 0};
	
	if(!_options)
		_ocd_create_arguments();
	_ocd_add_arg(o, _options_size);	
}

void ocd_usage()
{
	option* op;
	for (op = &_options[0]; op->type; ++op)
		printf("%s\n", optsusage(op));
}

void ocd_init(int* argc, char*** argv, ocd_requirements* reqs)
{
	ocd_parse(argc, argv);
	ocd_check_requirements(reqs);
	#ifdef ENABLE_TIMER
	TIMER_INIT;
	#endif
}

void ocd_finalize()
{
	#ifdef ENABLE_TIMER
	TIMER_STOP
	TIMER_PRINT
	TIMER_DEST
	#endif
}

void ocd_print_device_info(cl_device_id device)
{
	cl_int err;
	cl_uint int_val;
	cl_bool bool_val;
	cl_ulong long_val;
	char char_array[256];
	size_t ret_size,size_val;
	cl_platform_id platform_val;
	cl_device_type devtype_val;

	err = clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(char)*256,char_array,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Name");
	printf("Device %d: Name=%s\n",device,char_array);

	err = clGetDeviceInfo(device,CL_DEVICE_PLATFORM,sizeof(cl_platform_id),&platform_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Platform ID");
	printf("Device %d: Platform=%d\n",device,platform_val);

	err = clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(cl_device_type),&devtype_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Type");
	printf("Device %d: Type=%d\n",device,devtype_val);

	err = clGetDeviceInfo(device,CL_DEVICE_VENDOR,sizeof(char)*256,&char_array,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Vendor");
	printf("Device %d: Vendor=%s\n",device,char_array);

	err = clGetDeviceInfo(device,CL_DEVICE_VENDOR_ID,sizeof(cl_uint),&int_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Vendor ID");
	printf("Device %d: Vendor ID=%d\n",device,int_val);

	err = clGetDeviceInfo(device,CL_DEVICE_VERSION,sizeof(char)*256,&char_array,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Version");
	printf("Device %d: Version=%s\n",device,char_array);

	err = clGetDeviceInfo(device,CL_DEVICE_AVAILABLE,sizeof(cl_bool),&bool_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Availability");
	printf("Device %d: Available=%d\n",device,bool_val);

	err = clGetDeviceInfo(device,CL_DEVICE_ENDIAN_LITTLE,sizeof(cl_bool),&bool_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Endianness");
	printf("Device %d: Little Endian=%d\n",device,bool_val);

	err = clGetDeviceInfo(device,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&long_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Global Mem Size");
	printf("Device %d: Global Mem Size =%u\n",device,long_val);

	err = clGetDeviceInfo(device,CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),&long_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Local Mem Size");
	printf("Device %d: Local Mem Size=%u\n",device,long_val);

	err = clGetDeviceInfo(device,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(cl_uint),&int_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Max Clock Frequency");
	printf("Device %d: Max Clock Frequency=%u\n",device,int_val);

	err = clGetDeviceInfo(device,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&int_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Max Compute Units");
	printf("Device %d: Max Compute Units=%u\n",device,int_val);

	err = clGetDeviceInfo(device,CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(cl_ulong),&long_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Max Constant Buffer Size");
	printf("Device %d: Max Constant Buffer Size=%u\n",device,long_val);

	err = clGetDeviceInfo(device,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&long_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Max Mem Alloc Size");
	printf("Device %d: Max Mem Alloc Size=%u\n",device,long_val);

	err = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&size_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Max Work Group Size");
	printf("Device %d: Max Work Group Size=%d\n",device,size_val);

	err = clGetDeviceInfo(device,CL_DEVICE_PROFILING_TIMER_RESOLUTION,sizeof(size_t),&size_val,&ret_size);
	check(err == CL_SUCCESS,"Error Querying Device Profiling Timer Resolution");
	printf("Device %d: Profiling Timer Resolution=%d\n",device,size_val);
}

cl_device_id GetDevice(int platform, int device, cl_int dev_type)
{
    cl_int err;
    cl_uint nPlatforms = 1;
    err = clGetPlatformIDs(0, NULL, &nPlatforms);
    CHECK_ERROR(err);

    if (nPlatforms <= 0) {
        printf("No OpenCL platforms found. Exiting.\n");
        exit(0);
    }
    if (platform < 0 || platform >= nPlatforms) // platform ID out of range
    {
        printf("Platform index %d is out of range. \n", platform);
        exit(-4);
    }
    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof (cl_platform_id) * nPlatforms);
    err = clGetPlatformIDs(nPlatforms, platforms, NULL);
    CHECK_ERROR(err);

    cl_uint nDevices = 1;
    char platformName[100];
    err = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, sizeof (platformName), platformName, NULL);
    CHECK_ERROR(err);
    printf("Platform Chosen : %s\n", platformName);

    // query devices
	err = clGetDeviceIDs(platforms[platform], dev_type, 0, NULL, &nDevices);
	if(err == CL_DEVICE_NOT_FOUND)
	{
		fprintf(stderr,"No Supported Device Found of Type %d. Falling back to CPU.\n",dev_type);
		dev_type = CL_DEVICE_TYPE_CPU;
		err = clGetDeviceIDs(platforms[platform], dev_type, 0, NULL, &nDevices);
	}
	CHECK_ERROR(err);

    if (nDevices <= 0) {
        printf("No OpenCL Device found. Exiting.\n");
        exit(0);
    }
    if (device < 0 || device >= nDevices) // platform ID out of range
    {
        printf("Device index %d is out of range. \n", device);
        exit(-4);
    }
    cl_device_id* devices = (cl_device_id *) malloc(sizeof (cl_device_id) * nDevices);
    err = clGetDeviceIDs(platforms[platform], dev_type, nDevices, devices, NULL);
    char DeviceName[100];
    err = clGetDeviceInfo(devices[device], CL_DEVICE_NAME, sizeof (DeviceName), DeviceName, NULL);
    CHECK_ERROR(err);
    printf("Device Chosen : %s\n", DeviceName);

    return devices[device];
}
