#!/bin/bash

#========[ + + + + Requirements + + + + ]========#
#SBATCH -A m2_datamining
#SBATCH -p smp
#SBATCH -c 2
#SBATCH -J discrete_exps
#SBATCH --time=0-08:00:00
#SBATCH --mem 128G

#========[ + + + + Environment + + + + ]========#
module load lang/Julia

#========[ + + + + Job Steps + + + + ]========#
srun julia --project=../. exp1_discrete.jl