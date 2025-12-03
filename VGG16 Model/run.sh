container="/data/containers/msoe-tensorflow-20.07-tf2-py3.sif"

# Command to run inside container
command="python model.py --data /data/train --batch_size 8 --epochs 15 --main_dir /home/ad.msoe.edu/ruizi/AITools/Final Project --augment_data true --fine_tune true"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} /usr/local/bin/nvidia_entrypoint.sh ${command}