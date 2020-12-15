#define RETURN_CHECK \
if (ret == CL_BUILD_PROGRAM_FAILURE) \
{ \
    size_t log_size; \
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); \
    char *log = (char *) malloc(log_size); \
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); \
    fprintf(stderr, "%s\n", log); \
    exit(ret); \
} \
else if (ret != 0) \
{ \
    fprintf(stderr, "%d\n", ret); \
    exit(ret); \
}

#define RC(x) \
ret = x; \
RETURN_CHECK