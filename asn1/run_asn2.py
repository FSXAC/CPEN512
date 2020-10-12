#!python3

# This Python script runs the script for gaussian elimination
# For various cases and collect median execution times
# Results are saved in output asn2_results text file

import subprocess
import re
import statistics

COMPILE_CMD = './pt_compile.sh'
RUN_CMD = './ref_pthread.out'

# Test params specify the matrix size and number of threads
LESS_THREADS = [2, 4, 8]
MORE_THREADS = [2, 4, 8, 16, 32, 64]

TEST_PARAMS = [
    (8, LESS_THREADS),
    (16, LESS_THREADS),
    (512, MORE_THREADS),
    (1024, MORE_THREADS),
    (2048, MORE_THREADS),
    (4096, MORE_THREADS)
]

NUM_TRIALS_PER_COMBINATION = 5
NUM_TOTAL_TRIALS = NUM_TRIALS_PER_COMBINATION * (2 * 3 + 4 * 6)

# Regex for interpreting shell output
re_serial = re.compile('SERIAL TIME=(.+) s')
re_parallel = re.compile('PARALL TIME=([0-9e\-\+\.]+) s')
re_mismatch = re.compile('MISMATCH=(\d+)\n')

# results dictionalry
results = dict()

# Main
if __name__ == '__main__':
    test_count = 0
    for n, test_num_threads in TEST_PARAMS:

        # Store serial temp results
        serial_batch_result = list()

        # Set up results
        results[str(n)] = dict()

        for num_threads in test_num_threads:

            # Store the parallel temp results
            parallel_batch_result = list()

            # Compile file
            _ = subprocess.check_output([COMPILE_CMD, str(n), str(num_threads)])

            # Run trials
            for _ in range(NUM_TRIALS_PER_COMBINATION):
                print('Testing N=%d Threads=%d (%.1f%% complete): ' % (n, num_threads, 100.0 * test_count / NUM_TOTAL_TRIALS), end='')
                run_output = subprocess.check_output([RUN_CMD]).decode('utf-8')

                if 'Error' not in run_output:
                    serial_time = float(re_serial.findall(run_output)[0])
                    parallel_time = float(re_parallel.findall(run_output)[0])

                    if (int(re_mismatch.findall(run_output)[0]) != 0):
                        print("\nError: mismatch detected!")
                        print(run_output)
                        exit(1)

                    serial_batch_result.append(serial_time)
                    parallel_batch_result.append(parallel_time)

                    print("Serial: %.6e, Parallel: %.6e, Speedup: %.2f" % (serial_time, parallel_time, serial_time / parallel_time))
                else:
                    print('\nError: error while running the file, skipping')
                    print(run_output)

                test_count += 1

            # Find median of parallel
            results[str(n)][str(num_threads)] = statistics.median(parallel_batch_result)
        
        # Find median of the serial result
        results[str(n)]['serial'] = statistics.median(serial_batch_result)

    # Write results to file
    for n, result in results.items():
        with open('asn2_results_%s.csv' % n, 'w') as out_file:
            print('case,time', file=out_file)
            for num_threads, time in result.items():
                print('%s,%.6e' % (num_threads, time), file=out_file)