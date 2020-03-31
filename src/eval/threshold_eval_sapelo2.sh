#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=8
#PBS -l walltime=2:00:00:00
#PBS -l mem=32gb

cd $PBS_O_WORKDIR

ml Python/3.6.4-foss-2018a

mkdir -p models/FINAL/threshold_smoothed_cv/${FOLD}/${THRESH}/
pipenv run python src/eval/segment_images.py \
                    models/FINAL/src/ \
                    models/FINAL/cross_validation/saves/CE05_CV${FOLD}_epoch800 \
                    models/FINAL/threshold_smoothed_cv/${FOLD}/${THRESH}/ \
                    data/processed/training/real/*.j* \
                    --threshold ${THRESH} \
                    --fold_cv ${FOLD_CV} \
                    --fold ${FOLD} \
