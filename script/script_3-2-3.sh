#!/bin/sh
### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J Script_3-2-3
### -- ask for number of cores (default: 1) --
#BSUB -n 20
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 8GB of memory per core/slot --
#BSUB -R "rusage[mem=8GB]"
### -- specify that we want the job to get killed if it exceeds 10 GB per core/slot --
#BSUB -M 10GB
### -- set wall time limit: hh:mm --
#BSUB -W 72:00
### -- set the email address --
#BSUB -u s202566@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Script_3-2-3_%J.csv
#BSUB -e Error_3-2-3_%J.err

### — Add modules necessary to execute the script
##module load <list of the modules the program needs>
module load gcc/10.3.0-binutils-2.36.1
module load python3/3.9.6
module load openblas/0.3.17
module load numpy/1.21.1-python-3.9.6-openblas-0.3.17
module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load pandas/1.3.1-python-3.9.6


# here follow the commands you want to execute
python3 Main.py 3 1 2 1 40 160 20 500 20