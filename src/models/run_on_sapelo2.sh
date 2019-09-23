#PBS -S /bin/bash
#PBS -q gpu_q
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=10:00:00:00
#PBS -l mem=6gb

# If a name is given, run locally, otherwise check to see if
# running as a cluster job, If so, act accordingly.
if [ "$#" -eq 1 ]; then
      RUN=$1
  elif [ -n "$PBS_JOBNAME" ]; then
        RUN=$PBS_JOBNAME
        cd $PBS_O_WORKDIR
  else
        echo "Please provide the run name: run.sh RUN"
        exit
fi

ml CUDA/9.0.176
ml cuDNN/7.0.4-CUDA-9.0.176
ml Python/3.6.4-foss-2018a

# Use this to start a new training
#pipenv run python train.py --comment $RUN --seed 1 --weighted-ce 0.05 --save-interval 100

# Use this to resume training from a save point
pipenv run python train.py --comment $RUN --resume ./saves/FINAL_SECOND_RUN/FINAL_CE05_epoch1800
