# Synthetic Taxi Trajectory Generation
This repository contains the code for generating synthetic taxi trajectories using CTGAN. The generation is based on the paper "CTGAN: A Conditional Generative Adversarial Network for Tabular Data" by Lei Xu et al. The code is implemented in Python using PyTorch.


## Data-Pipeline
1. Data Cleanup - gen_taxi_trajectoriers/modelling/data_prop.py
2. Data Synthesis - gen_taxi_trajectoriers/modelling/model_ctgan.py
3. Post-Processing - gen_taxi_trajectoriers/modelling/gps_path.py
4. Evaluation - notebooks/evaluation.py
