# ml_visual_reasoning
ML 7641 Project Team 17 (Fall 2021)

# Run training script
- ensure you have the "pkl_files" folder placed in the repo folder on your local machine
- setup the conda environemnt using ml_project_conda.yaml file and activate it
- Run the following command: python train_dataloader.py --num_epochs 10 --batch_size 128 --learning_rate 1e-6 --train_samples 10000 --val_samples 3108 --tensor_board_viz yes

# Tensorboard Visualisation
- if you ran the train_dataloader.py with tensor_board_visualisation argument set to "yes", a new log folder will be created in the "runs" folder present in the repository
- in the "runs" directory, run the follwoing command: tensorboard --logdir *name of log folder* --port=6006

# GPU Usage Monitoring
- you can monitor the GPU usage by running "nivida-smi" in terminal
- once the trainign script has run, if the GPU memory is not freed up, you can clear it by running "nvidia-smi" and checking the process PID which uses the GPU and then run the command: kill -9 *Process PID*
