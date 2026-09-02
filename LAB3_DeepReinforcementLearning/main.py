import os
import sys
import gymnasium as gym  
import torch
from utils.helpers import PolicyNet, execute_experiment, ValueNet, PolicyNetDeeper, ValueNetDeeper
from utils.helpers import a2c_train, plot_running_rewards, plot_validation_rewards
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def main():

    device = torch.device("cpu")
    print("Running Device: {}".format(device))

    config_path = "config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    EXE_TO_BE_RUN = config["EXE_OPTS"]["run_exercise"]
    DEFAULT_SEED = 200  
    for exe in EXE_TO_BE_RUN:
        if exe == 1:
            print("========== Executing exercise 1 ========== ")
            gammas = config["EXE1"]["gammas"]
            scale_factors = config["EXE1"]["scale_factors"]
            n = config["EXE1"]["n"]
            m = config["EXE1"]["m"]
            NUM_EPISODES = config["EXE1"]["NUM_EPISODES"]

            experiments_results_paths = list()
            for g in gammas:
                for sf in scale_factors:
                    set_seed(DEFAULT_SEED)
                    experiment_name = "cartpole_gamma_{}__sf_{}".format(g, sf)
                    print("***\n")
                    print("Running Experiment: {}".format(experiment_name))
                    print("***\n")
                    curr_exp_path = execute_experiment(experiment_name=experiment_name, env_name="CartPole-v1", 
                                    gamma=g, num_episodes=NUM_EPISODES, scale_factor=sf, env_render=None, n = n, m = m)
                    experiments_results_paths.append(curr_exp_path)
            print("=== Exercise 1 is finished ===\n")
            print("=== You can find all its results in {}\n\n".format(experiments_results_paths))

        if exe == 2:
            print("\n\n\n========== Executing exercise 2 ========== ")
            PATH_TO_EXE2 = "excercise2"
            NUM_EPISODES = config["EXE2"]["NUM_EPISODES"]
            
            experiment_name_ns = "no_subtract"
            full_name_exp = os.path.join(PATH_TO_EXE2, experiment_name_ns)
            set_seed(DEFAULT_SEED)
            experiment_name_ns_path = execute_experiment(full_name_exp, num_episodes= NUM_EPISODES)
            print("REINFORCE experiment without subtracting average return finished! Results at: {}".format(experiment_name_ns_path))

            experiment_name_s = "subtract"
            full_name_exp = os.path.join(PATH_TO_EXE2, experiment_name_s)
            set_seed(DEFAULT_SEED)
            experiment_name_s_path = execute_experiment(full_name_exp, num_episodes= NUM_EPISODES, subtract_sb=True)
            print("REINFORCE experiment subtracting average return finished! Results at: {}".format(experiment_name_s_path))

            print("\n---\n")
            print("\n--- Executing REINFORCE with Value Function ---")
            experiment_name_vf = "value_function"
            full_name_exp = os.path.join(PATH_TO_EXE2, experiment_name_vf)
            set_seed(DEFAULT_SEED)
            env = gym.make("CartPole-v1")
            value_net = ValueNet(env)
            env.close()

            experiment_name_vf_path = execute_experiment(experiment_name=full_name_exp,env_name="CartPole-v1",num_episodes=NUM_EPISODES,val_nn=value_net)

            print("REINFORCE experiment with Value Function finished! Results at: {}".format(experiment_name_vf_path))
            print("\n=== Exercise 2 is finished ===\n")

        if exe == 3:
            SEEDS = [42, 99, 123]
            NUM_EPISODES = 1000
            SCALE_FACTOR_CARTPOLE = 0.02
            SCALE_FACTOR_LUNAR = 0.45

            # 1. CartPole A2C
            print("\n========== Executing Exercise 3: A2C (CartPole) ========== ")
            BASE_EXE3_CARTPOLE = os.path.join("exercise3", "cartpole_multi_run")

            for seed in SEEDS:
                print(f"\n- Starting CartPole Run with Seed: {seed} <---")
                set_seed(seed)
                seed_output_dir = os.path.join(BASE_EXE3_CARTPOLE, f"run_seed_{seed}")
                
                env = gym.make("CartPole-v1")
                env.action_space.seed(seed)
                actor_net = PolicyNet(env).to(device)
                critic_net = ValueNet(env).to(device)

                running_rewards, eval_results = a2c_train(
                    output_dir=seed_output_dir, actor=actor_net, critic=critic_net,
                    env=env, seed=seed, num_episodes=NUM_EPISODES, gamma=0.99, n=50, m=10, scale_factor=SCALE_FACTOR_CARTPOLE
                )
                env.close()
                
                plot_running_rewards(running_rewards, mode=1, experiment_dir=seed_output_dir)
                plot_validation_rewards(eval_results, experiment_dir=seed_output_dir)

            # Lunars A2C
            print("\n========== Executing Exercise 3: A2C (Lunars Standard) ========== ")
            BASE_EXE3_LUNARS = os.path.join("exercise3", "lunars_multi_run")

            for seed in SEEDS:
                print(f"\n Starting Lunars Standard Run with Seed: {seed} ")
                set_seed(seed)
                seed_output_dir = os.path.join(BASE_EXE3_LUNARS, f"run_seed_{seed}")
                
                env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                            enable_wind=False, wind_power=15.0, turbulence_power=1.5)
                env.action_space.seed(seed)
                actor_net = PolicyNet(env).to(device)
                critic_net = ValueNet(env).to(device)

                running_rewards, eval_results = a2c_train(
                    output_dir=seed_output_dir, actor=actor_net, critic=critic_net,
                    env=env, seed=seed, num_episodes=NUM_EPISODES, gamma=0.99, n=50, m=10, scale_factor=SCALE_FACTOR_LUNAR
                )
                env.close()
                
                plot_running_rewards(running_rewards, mode=1, experiment_dir=seed_output_dir)
                plot_validation_rewards(eval_results, experiment_dir=seed_output_dir)

            # Lunars Deeper A2C
            print("\n========== Executing Exercise 3: A2C (Lunars Deeper) ========== ")
            BASE_EXE3_LUNARS_DEEPER = os.path.join("exercise3", "lunars_Deeper_multi_run")

            for seed in SEEDS:
                print(f"\n Starting Lunars Deeper Run with Seed: {seed} ")
                set_seed(seed)
                seed_output_dir = os.path.join(BASE_EXE3_LUNARS_DEEPER, "run_seed_{}".format(seed))
                
                env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                            enable_wind=False, wind_power=15.0, turbulence_power=1.5)
                env.action_space.seed(seed)
                actor_net = PolicyNetDeeper(env).to(device)
                critic_net = ValueNetDeeper(env).to(device)

                running_rewards, eval_results = a2c_train(
                    output_dir=seed_output_dir, actor=actor_net, critic=critic_net,
                    env=env, seed=seed, num_episodes=NUM_EPISODES, gamma=0.99, n=50, m=10, scale_factor=SCALE_FACTOR_LUNAR
                )
                env.close()
                
                plot_running_rewards(running_rewards, mode=1, experiment_dir=seed_output_dir)
                plot_validation_rewards(eval_results, experiment_dir=seed_output_dir)

if __name__ == "__main__":
    main()