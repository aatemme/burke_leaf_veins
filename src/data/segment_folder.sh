#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=8
#PBS -l walltime=2:00:00:00
#PBS -l mem=32gb
#PBS -N segment_folder
cd $PBS_O_WORKDIR

ml Python/3.6.4-foss-2018a

pipenv run python src/eval/segment_images.py \
                    ${model} \
                    ${state} \
                    ${results} \
                    ${images} \
                    --threshold ${THRESH} \
