#!/bin/bash
#PBS -q ccs_short
#PBS -l walltime=6:00:00 
#PBS -l nodes=1:eightcore:ppn=3 
#PBS -m ae
#PBS -d /scratch03/bcummins/polymercode/trunk/pythoncode

time python vecode/mainC.py 

echo End Job
