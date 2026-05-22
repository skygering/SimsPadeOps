#!/bin/bash
#SBATCH -J modal_analysis_data           # Job name
#SBATCH -o /scratch/10264/sgering/Data/F_0013_X_SU_PI_Files/modal_analysis_data.o%j        # Name of stdout output file
#SBATCH -e /scratch/10264/sgering/Data/F_0013_X_SU_PI_Files/modal_analysis_data.e%j        # Name of stderr error file
#SBATCH -p skx                       # Queue (partition) name
#SBATCH -N 1             # Total # of nodes
#SBATCH --ntasks-per-node 48         # Total # of cores
#SBATCH -t 3:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-user=sgering@mit.edu  # Email ID  
#SBATCH --mail-type=all              # Send email at begin and end of job
#SBATCH -A TG-ATM170028              # Allocation details
#SBATCH --exclude=c476-092

export FI_PROVIDER=psm2

# Launch MPI code...
python /scratch/10264/sgering/SimsPadeOps/analysis/modal_analysis_exploration.py