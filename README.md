# Energy Harvesting in RIS-UAV system using Deep Reinforcement Learning
## Requirements
- Python: 3.6.13
- Pytorch: 1.10.1
- gym: 0.15.3
- numpy: 1.19.2
- matplotlib
- pandas
- Stable-Baselines3

## Usage
#### Descriptions of folders
- The folder "XXXX-MultiUT-Time" is the source code for the time-domain EH scheme in the multiple user scenario.
- The folder "XXXX-MultiUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the multiple user scenario.
- The folder "XXXX-SingleUT-Time" is the source code for the time-domain EH scheme in the single user scenario.
- The folder "XXXX-SingleUT-Two" is the source code for the two-domain (Time and Space) EH scheme in the single user scenario.
- The folder "CreateData" is the source code for generating dataset of trajectories files for users and the UAV.

#### Descriptions of files
- For the Exhaustive Algorithm, the communication environment is impletemented in 'ARIS_ENV.py'.
- For DRL-based algorithms, the communication environment is impletemented in 'gym_foo/envs/foo_env.py'.
- You can change the dataset and the scenario in 'gym_foo/envs/foo_env.py'.

#### Training phase
1. For the TD3 and DDPG, please execute the TD3.py and DDPG.py to train the model, such as
```
python TD3.py / python DDPG.py
```
***Please change the training mode in the file "gym_foo/envs/foo_env.py" before you executing the training progress.***
For example:
```
class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, LoadData = True, Train = False, multiUT = True, Trajectory_mode = 'Fermat', MaxStep = 41):        
```
If you want to conduct the training phase, the value of "Train" should be "True", otherwise, the value of "Train" should be "Flase" when excuting the testing phase.

2. For the exhaustive search, please execute the ExhaustiveSearch.py to reproduce the simulation results.
3. For the SD3, please execute main.py to train a new model. 

***Please use the version of 0.15.3 for Gym, otherwise there may have some issues in the training phase.***

#### Testing phase
Please execute test.py to evaluate DRL models. Before you produce the testing results, please change the dataset and scenario in 'gym_foo/envs/foo_env.py'.

#### The EH efficiency
The EH efficiency = the harvested energy / the received energy from RF signals
