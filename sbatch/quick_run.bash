#!/bin/bash
#SBATCH -p alldlc_gpu-rtx2080 # partition
##SBATCH --mem 4G # memory pool for each core (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -c 8 # number of cores                                                         
#SBATCH --gres=gpu:8
#SBATCH -D /home/huang/hcg/projects/nerf_manipulation/code/nerf_manipulation/
#SBATCH -o /work/dlclarge1/huang-nerf/logs/%x.%N.%j.out # STDOUT
# #SBATCH -e /work/dlclarge1/huang-nerf/errs/%x.%N.%j.err # STDERR
#SBATCH -J nerf_manipulation # sets the job name. If not specified, the file anme will be used as job name
#SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job    )

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";


# shellcheck disable=SC1091
source ~/.bashrc                                                                                                                                                                         
conda activate pixelnerf

srun -u \
python -u /home/huang/hcg/projects/nerf_manipulation/code/nerf_manipulation/train/train.py \
    -n nerf_rl \
    -D /work/dlclarge2/meeso-lfp/nerf_play_dataset_processed/ \
    -c /home/huang/hcg/projects/nerf_manipulation/code/nerf_manipulation/conf/exp/real_world.conf \
    -V 1 \
    -B 12 \
    --gpu_id='0 1 2 3 4 5 6 7' \
    --visual_path /work/dlclarge1/huang-nerf \
    --resume

# print information about the end-time
echo "DONE";
echo "Finished at $(date)";
                              
