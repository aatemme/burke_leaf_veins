## This script can be used to start juptyer on an interactive node on sapelo2
# and allow connections from the remote (your) computer.  
# see https://wiki.gacrc.uga.edu/wiki/Jupyter-Sapelo2 for details
NOTEBOOKPORT=9465

IPUSED=$(hostname -i)

echo "NOTEBOOKPORT is " $NOTEBOOKPORT

echo "IPUSED is " $IPUSED

ml Python/3.6.4-foss-2018a

pipenv run jupyter lab --port $NOTEBOOKPORT --ip=$IPUSED --no-browser
