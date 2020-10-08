- https://computing.llnl.gov/tutorials/pthreads/#Overview
    - “MPI libraries usually implement on-node task communication via shared memory, which involves at least one memory copy operation (process to process).”
    - “For Pthreads there is no intermediate memory copy required because threads share the same address space within a single process. There is no data transfer, per se. It can be as efficient as simply passing a pointer.”
    
- https://computing.llnl.gov/tutorials/parallel_comp/
- https://blog.albertarmea.com/post/47089939939/using-pthreadbarrier-on-mac-os-x
    - Barriers are not implmeneted by default in Mac