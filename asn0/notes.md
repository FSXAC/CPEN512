- Using scanf to get the N size is not ideal because compiler cannot do optimization based on N size before hand
- If the computed matrix is not used anywhere in the code, the optimizer may optimize the calculation away — resulting in very small clock ticks (this is because the calculation was never carried out).

### Finding Custom Options

I’ve set up the program to run -O3 and extra flags each for 200 trials.

For existing “extra” config, the average clock ticks is **17577**.

- Using just -O3 gets me **16756** so lets start here
- 

