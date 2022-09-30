# CSC4005-FA22

## Slurm notes
```
# intereactive session with 1 node, 8 cores, 5 minutes, with partition debug/project
salloc -N1 -n8 -t5 -p Debug
salloc -N1 -n8 -t5 -p Project

# submit & cancel jobs
sbatch job.sh
scancel $JOBID
scancel --user=$(whoami)

# view queue
squeue

# submission history
sacct

# view node info
sinfo -a
```

## CMake Notes

```
# configure in build folder
cmake -B build
# build
cmake --build build
```




