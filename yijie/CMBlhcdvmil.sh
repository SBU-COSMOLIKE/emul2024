#!/bin/bash
#SBATCH --job-name=cmblhcdv
#SBATCH --output=./output/cmblhcdv.txt
#SBATCH --time=8:00:00
#SBATCH -p hbm-large-96core
#SBATCH --nodes=30
#SBATCH --ntasks=1200
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1


echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

cd $SLURM_SUBMIT_DIR

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core python ./CMBlhcdvmil.py\
      -f ${SLURM_ARRAY_TASK_ID}