# ml_visual_reasoning
ML 7641 Project Team 17 (Fall 2021)

# Run training script
- ensure you have the "pkl_files" folder downloaded on your local machine
- setup the conda environemnt using ml_project_conda.yaml file and activate it
- Run the following command (values can be different depending on your choice of hyperparameters): python train_dataloader.py --num_epochs 10 --batch_size 128 --learning_rate 1e-6 --train_samples 10000 --val_samples 3108 --path_pkl_files /home/yusuf/Desktop/Georgia\ Tech/Fall\ 2021/CS7641/project_midterm/API/pkl_files --tensor_board_viz yes
- --path_pkl_files will be different depending on the absolute lcoation of the pkl_files directory on your local machine

# Tensorboard Visualisation
- if you ran the train_dataloader.py with tensor_board_visualisation argument set to "yes", a new log folder will be created in the "runs" folder present in the repository which will have the train and validation loss values from your experimental run
- in the "runs" directory, run the following command: tensorboard --logdir *name of log folder* --port=6006

# GPU Usage Monitoring
- you can monitor the GPU usage by running "nivida-smi" in terminal
- once the trainign script has run, if the GPU memory is not freed up, you can clear it by running "nvidia-smi" and checking the process PID which uses the GPU and then run the command: kill -9 *Process PID*
