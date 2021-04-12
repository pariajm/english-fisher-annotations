#!/bin/bash

#SBATCH --time=03:30:00
#SBATCH --mem=25GB
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=paria.jamshid-lou@students.mq.edu.au 
#SBATCH --output=cp.out
#SBATCH --error=cp.err

#module load python/3.6.1
#module load pytorch/1.6.0-py36-cuda10-mpi
#module load cuda/10.2.89
#module load cudnn/v8.0.5-cuda102

module load python/3.6.1
module load pytorch/1.0.0-py36-cuda90
module load cuda/9.0.176
module load cudnn/v7.1.3-cuda90
#module load apex/0.1
#export PYTHONPATH="$PYTHONPATH:/home/jam212/.local/lib/python3.6/site-packages"


export PYTHONPATH="$PYTHONPATH:/home/jam212/.local/lib/python3.6/site-packages"

#UDA_VISIBLE_DEVICES=0,1 python main_valid.py --input-path /scratch1/jam212/interspeech/raw_swbd/ --output-path /scratch1/jam212/interspeech/tag_gold --model ./model/swbd_fisher_bert_Edev.0.9078.pt 

CUDA_VISIBLE_DEVICES=0,1 python main_valid.py --input-path  /scratch1/jam212/asr/trans/manifest-out/data_switchboard/train_label/  --output-path /scratch1/jam212/asr/trans/manifest-out/data_switchboard --model ./model/swbd_fisher_bert_Edev.0.9078.pt
