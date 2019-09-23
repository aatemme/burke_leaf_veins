#PBS -S /bin/bash
#PBS -q batch
#PBS -l nodes=1:ppn=8
#PBS -l walltime=5:00:00
#PBS -l mem=32gb
#PBS -N segment_folder
cd $PBS_O_WORKDIR

###########################
#
# This script can be used to extract the vein lengths from an image
# within a folder
#
# To submit, run `qsub segment_extract_length_folder_sapleo2.sh`
#
# Segmentation uses a lot of RAM, hence the request for 32gb
#
# By default, this script requests 5 hours. It takes about 90 seconds per
# image for segmentation. The walltime could be adjusted to the number of
# images being segemented.
#
# The script places all results in the RESULTS_FOLDER. This includes:
#       The extracted lengths in `RESULTS_FOLDER/lengths.csv`
#       The segmentation probabilities as P * 2^16 as a uint16.
#       The segmentation mask overplayed on the image for visual
#       inspection.
#
# NOTE:
#   The progress is written to the error sapelo output file (segment_folder.e*)
#
###########################

ml Python/3.6.4-foss-2018a


#These need to be relative to the project root.
# The base folder is whereever the script is subitted to qsub from.
PROJECT_ROOT=../..
FINAL_MODEL=$PROJECT_ROOT/models/FINAL/src/
FINAL_STATE=$PROJECT_ROOT/models/FINAL/saves/FINAL_CE05_epoch1900 # <- This is the final model weights
THRESHOLD=0.6 #<- This was chosen using cross validation. It does not need changed
RESULTS_FOLDER=$PROJECT_ROOT/data/interm/exmaple_run

#This input supports wildcards, in this case, it tells the script to
# segment all (*) images in the real folder that end in .jpeg
IMAGES=$PROJECT_ROOT/data/processed/validation/real/*.jpeg

mkdir -p $RESULTS_FOLDER
pipenv run python $PROJECT_ROOT/src/data/segment_extract_length_folder.py \
                    $FINAL_MODEL \
                    $FINAL_STATE \
                    $RESULTS_FOLDER \
                    $IMAGES \
                    --threshold $THRESHOLD \
