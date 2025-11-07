#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=output.out

source /vol/csedu-nobackup/course/I00041_informationretrieval/users/vbolzonella/IR_project/.venv/bin/activate

cd /vol/csedu-nobackup/course/I00041_informationretrieval/users/vbolzonella/IR_project

python3 main.py --queries '/vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/IR_project/data/longfact-objects_celebrities.jsonl' --index '/vol/csedu-nobackup/course/I00041_informationretrieval/users/analeopold/IR_project/indexes/wiki_dump_index'
