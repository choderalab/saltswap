# Benchmarks for `updateParametersInContext`
Summary for timings of `run_benchmarks.sh` for water box length=30 Angs
and various platforms. This results indicates that on GPUs, `updateParametersInContext` 
runs ~4 times slower in OpenMM 7.2 than 7.1.

## GTX Titan, CUDA 8.0
####Python 3.5, OpenMM 7.1
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.024    0.000    7.078    0.000 openmm.py:8705(updateParametersInContext)
```
####Python 3.5, OpenMM 7.2
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.024    0.000   30.505    0.002 openmm.py:8705(updateParametersInContext)
```
####Python 2.7, OpenMM 7.1
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.023    0.000    7.241    0.000 openmm.py:8705(updateParametersInContext)
```
####Python 2.7, OpenMM 7.2
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.027    0.000   30.655    0.002 openmm.py:8705(updateParametersInContext)
```
## CPU (iMac, 3.2 GHz Intel Core i5)
####Python 3.5, OpenMM 7.1
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.016    0.000   10.123    0.001 openmm.py:16121(updateParametersInContext)
```
####Python 3.5, OpenMM 7.2
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.016    0.000   10.189    0.001 openmm.py:16121(updateParametersInContext)
```
####Python 2.7, OpenMM 7.1
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.016    0.000    9.449    0.000 openmm.py:16121(updateParametersInContext)
```
####Python 2.7, OpenMM 7.2
```
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
20000    0.017    0.000   10.154    0.001 openmm.py:16121(updateParametersInContext)
```
