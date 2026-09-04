# Getting Up to Speed with Reinforcement Learning

Author: Edoardo Canti


## About the Virtual Environment:
This LAB has a different *conda* environment (DRL) wrt the one provided in flipped lecture. Requirements are reported in `requirements.txt`

## Organization
### Directories and Files:
0. `requirements.txt`: requirements for the used conda environment (Lunars Environment requested an additional package with respect to the original environment).
1. `config.yaml`: provides config params for the execution of the main file (which automatically executes everything)
2. `utils/` contains several function that will be used throughout the designing process:
    * `helpers.py`: a single file containing all functions used throughout the code;
    * `legacy.py`: the original flipped lecture code (used as basis for the implemented functions)
3. `cartpole_gamma_<val1>__sf_<val2>`: each of these directory contains plots and jsons for excercise 1. `<val1>` is taken from `config.yaml EXE1::gammas` and `<val2>` is taken from `EXE1::scale_factors` 
4. `exercise2`: Contains a directory for each sub exercise REINFORCE with NO SUBTRACT, with SUBTRACT and with VALUE_FUNCTION.
5. ``exercise3``: Contains three different subdirs one involving Cart Pole multirun, another for Lunars Lander multirun and the last one for Lunars Lander but using a more "complex" architecture for both actor and critic netowrks

## Execution Advice
  After install all necessary dependencies you can proceed into the project directory, activate the environment and run the python main script:
  ```
> cd LAB3_DeepReinforcementLearning
> conda activate DLR
> (DLR) $ python -m main.py
```
  
  The `config.yaml` is set by default to execute all exercises in sequence
  
  <br>
  

## Some insights
For this laboratory several experiments have been conducted. <br>
Cartpole environment has been executed with several gamma values (that you can find in the config file) and 2 distinct values for **Scale Factor**.<br>
<br>

**Scale Factor** is a variable introduced in this project in order to monitor how the agent learns during the training.<br>
More detailed infos about the "idea" behind scale factor can be found in the `DLA_FULL_REPORT.pdf`, you can think to this value as a scaler of the original **Observation Variables** domain, that "triggers" a counter whenever the extremal value of the scaled domain is reached. This is done in order to evaluate how the agent changes its moves while learning (ideally this value is in (0,1]), the greater it is the larger domain margin is allowed).<br>
by looking at the training plots you can see the values reached by each observation variable, at each step, by looking at the background heatmap (for further details I encourage you to take a look at the report).
