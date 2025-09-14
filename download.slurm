#!/bin/bash
#SBATCH --job-name=download_calvin
#SBATCH --partition=general          # partition name
#SBATCH --qos=normal                 # qos
#SBATCH --time=24:00:00              # time (48h)
#SBATCH --output=outcome_%j.out      # Standard output file (%j is job ID)
#SBATCH --error=outcome_%j.err       # Standard error file (%j is job ID)

cd ~/calvin/dataset
echo "Running in: $(pwd)"
sh download_data.sh D


# # Forward WANDB API key into container
# export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
# # Forward Qt offscreen rendering variable
# export SINGULARITYENV_QT_QPA_PLATFORM=offscreen

# # Run inside Apptainer container
# xvfb-run -a apptainer exec \
#     --nv \
#     --fakeroot \
#     --writable-tmpfs \
#     --bind /etc/ssl/certs:/etc/ssl/certs:ro \
#     --bind /etc/pki:/etc/pki:ro \
#     --bind /data/user_data/sreyasv/:/workspace \
#     --bind /home/sreyasv/Projects/wmsrl:/home/sreyasv/Projects/wmsrl \
#     /home/sreyasv/Projects/wmsrl/wmrl.sif \
#     bash -c "cd agent/flowdiffusion && pip install -e . && cd .. && cd .. && python -m agent.call_diff_agent_rl_bench"
