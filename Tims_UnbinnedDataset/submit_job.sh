#!/bin/bash -l

### before submitting change results --> results_job in make.py line 98 + 128 for different output folder

#SBATCH --job-name=unbinned-sensitivity
### Time your job needs to execute, e. g. hh:mm:ss
#SBATCH --time=03:10:00
## #SBATCH --output=./out/     # not recommended for RRZE clusters
## #SBATCH --mem=16G
### for clean working environment
#SBATCH --export=NONE 
unset SLURM_EXPORT_ENV  # after that no more SBATCH commands can be defined

### The last part consists of regular shell commands:
conda activate gammapy-1.0
### Change to working directory
cd /home/hpc/caph/mppi086h/gammapy-unbinned-analysis/EventDataset/

### Run your parallel application
python /home/hpc/caph/mppi086h/gammapy-unbinned-analysis/Tims_UnbinnedDataset/mc_sensitivity.py 0.02 24

### type in shell: sbatch submit_job.sh