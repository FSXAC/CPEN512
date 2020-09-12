- Using scanf to get the N size is not ideal because compiler cannot do optimization based on N size before hand
- If the computed matrix is not used anywhere in the code, the optimizer may optimize the calculation away — resulting in very small clock ticks (this is because the calculation was never carried out).

### Finding Custom Options

I’ve set up the program to run -O3 and extra flags each for 200 trials.

For existing “extra” config, the average clock ticks is **17577**.

- Using just -O3 gets me **16756** so lets start here
  - This could be the `march` and `mtune` flags set to *barcelona*
  - `less /proc/cpuinfo` shows that CPEN512 is using AMD Opteron Processor 6128 and there might be a flag `opteron-sse3` that could be interseting to look at.

For the above test, I repeated it on CPEN512 computer.

- Using extra config (with `march` and `mtune` set to *barcelona*) gives **18823**
- Using just -O3 gives **19646**

So yes, setting the CPU type do help with optimization.

- Trying `march` and `mtune` of *opteron-sse3* gives **19452** — not good and revert.
- Setting `-m32` slowed things way down: **69129**

So staying with the extra options, but modifying it:

- Changing O3 to OFast: **18821** (not significantly faster but could cause instability)
- Adding -funsafe-loop-optimizations: 18812
- Using march=native: 18814
- Adding -ftree-vectorize: 18812
- Adding -msse -msse2 -msse3 -mmmx -m3dnow: 18809

I think I’ll stop here.

---

### Tiling Matrix Multiplication

I’m following the algorithm on [Wikipedia](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Divide_and_conquer_algorithm).

- Using `sysctl -a` on my Mac, I can see that my Macbook has a cache line size of 64, and cache size of 256. But system report gives 256 KB of L2 Cache per core.
- On cpen512.ece, thisthe cache size is given as 512 KB per core. It also shows cache_alignment to be 64. 
- So, that gives us 8192 cache lines.

Ideally, the matrix should be divided into tiles of $\sqrt{M}$ by $\sqrt M$, where $M$ is size of the cache. 

Using this idea, the idea tile size for the ECE computer is 512.

