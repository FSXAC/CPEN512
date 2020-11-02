# This compiles and runs the files

import subprocess
import re
import statistics

COMPILE_CMD = './compile'
COMPILE_CUDA_ONLY_CMD = './compile_cuda_only'
RUN_CMD = './ref.out'

# Test params specify the matrix size
MAT_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Different thread block sizes (fixed on N=4096, default 1024)
BLOCK_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

re_serial = re.compile('SERIAL TIME=(.+) s')
re_parallel = re.compile('PARALL TIME=(.+) s')

results_mat = dict()
results_bs = dict()

NUM_TRIALS_PER_COMBINATION = 5

if __name__ == '__main__':
    
    # Mat size test
    for n in MAT_SIZES:
        _ = subprocess.check_output([COMPILE_CMD, str(n), str(1024)])

        print('Testing N=%d' % n)
        
        serial_result = list()
        parallel_result = list()

        for _ in range(NUM_TRIALS_PER_COMBINATION):
            run_output = subprocess.check_output([RUN_CMD]).decode('utf-8')

            if 'Error' not in run_output:
                try:
                    serial_time = float(re_serial.findall(run_output)[0])
                    parallel_time = float(re_parallel.findall(run_output)[0])
                except IndexError as identifier:
                    print('\nError: matching error!')
                    print(run_output)
                    continue

            serial_result.append(serial_time)
            parallel_result.append(parallel_time)

        # For each result, get the median
        results_mat[n] = (
            statistics.median(serial_result),
            statistics.median(parallel_result)
        )
    
    # Block size test
    for bs in BLOCK_SIZES:
        _ = subprocess.check_output([COMPILE_CUDA_ONLY_CMD, str(4096), str(bs)])
        print('Testing BS=%d' % bs)

        parallel_result = list()

        for _ in range(NUM_TRIALS_PER_COMBINATION):
            run_output = subprocess.check_output([RUN_CMD]).decode('utf-8')

            if 'Error' not in run_output:
                try:
                    parallel_time = float(re_parallel.findall(run_output)[0])
                except IndexError as identifier:
                    print('\nError: matching error!')
                    print(run_output)
                    continue

            parallel_result.append(parallel_time)
    
        results_bs[bs] = statistics.median(parallel_result)
    
    # write results to file
    with open('size_test.csv', 'w') as out:
        print('N,serial,parallel', file=out)
        
        for n, result in results_mat.items():
            serial_time, parallel_time = result
            print('%d,%6e,%6e' % (n, serial_time, parallel_time), file=out)

    with open('bs_test.csv', 'w') as out:
        print('BS, time', file=out)

        for n, result in results_bs.items():
            print('%d,%6e' % (n, result), file=out)


