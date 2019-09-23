#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=8
#PBS -l walltime=2:00:00:00
#PBS -l mem=8gb

cd $PBS_O_WORKDIR

ml Python/3.6.4-foss-2018a

# There was a disk i/o error if I tried to write the data from each
# job into the same sqlite file
pipenv run python src/eval/extract_vein_length.py \
                        sqlite:///${OUTPUT_FILE} \
                        vein_lengths \
                        ${THRESH} ${INPUT_FILES}
