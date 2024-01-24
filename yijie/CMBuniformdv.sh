#!/bin/bash
#SBATCH --job-name=cmbunidv
#SBATCH --output=./output/cmbuni2.txt
#SBATCH --time=48:00:00
#SBATCH -p long-96core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G


echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1



echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR


export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

python ./CMBuniform_${SLURM_ARRAY_TASK_ID}.py
