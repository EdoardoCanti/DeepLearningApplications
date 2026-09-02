import torch
import numpy as np
import torch.nn as nn
import json
from torch.distributions import Categorical
import os
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# The output of env.reset() is a "tupled" array

# This link provide documentation for accessing that gymnasium::BOX object type
# https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box
# I want low and high because they represent lower and upper bounds respectively for each observation variable in 
# the observation space

# scale_factor has been added after the first impelementation
# I think it would be nice to have an idea of how much the agent approaches to the bounds (upper or lower)
# for each variable in the obserbation space.
# e.g: 
# https://gymnasium.farama.org/environments/classic_control/cart_pole/ 
# shows observation space for cart pole, the first variable is the cart position
# If I scale by a certain amount and keep track of how many times the variable reach that scaled value
# I have an idea of how many times the agent came close to an error
# Ideally the scale factor should be in (0,1]; 1 is default because means (no scale factor: mantains original limits)
# 
def get_observation_variables_domains(environment, scale_factor=1.0) -> dict:
    if not environment.observation_space:
        raise ValueError("Passed argument doesn't have an observation space. Please provide a valid environment")
    lower_bounds = environment.observation_space.low
    upper_bounds = environment.observation_space.high
    # Limits lists must have the same len
    if len(lower_bounds) != len(upper_bounds):
        raise ValueError("This Environment has a different number lower bounds and upper bounds. Cannot proceed")
    observation_variables_domains = dict()
    for i in range(len(lower_bounds)):
        min_val = lower_bounds[i]
        max_val = upper_bounds[i]
        if min_val != -np.inf: # cart pole showed that some limit are infinite (no limits)
            min_val *= scale_factor
        if max_val != np.inf:
            max_val *= scale_factor
        observation_variables_domains[i] = {"min": min_val, "max": max_val}
    return observation_variables_domains


# now given the env.reset output it would be nice to have something that compares the final value with the domains
def evaluate_observations_bounds(environment, env_output: tuple, scale_factor: float = 1.0):
    #get the observation space var domains
    ovd = get_observation_variables_domains(environment, scale_factor=scale_factor) 
    env_output = env_output[0] # I need to untuple that because of (first comment at starting)

    # iterate over the indices for each realization of all the variables in the observation space
    # the result will be a dictionary itself with {<variable>:{is_out_domain; error_magnitude }} (error is probably not the right term sorry)
    results = dict()
    for i in range(len(env_output)):
        val = float(env_output[i]) # I had to convert this to otherwise receive error
        min_val = float(ovd[i]["min"])
        max_val = float(ovd[i]["max"])
        results[i] = {
            # if the output val is inside the domain set is_out_domain as false
            "is_out_domain": bool(not (min_val <= val <= max_val)), 
            # now if the val is lower than the lower limit the discard is negative 
            "error_magnitude": float(
                min_val - val if val < min_val 
                else (val - max_val if val > max_val else 0.0) # else is positive
            )
        }
    return results


def select_action(env, obs, policy, deterministic=False):
    probs = policy(obs)
    dist = Categorical(probs)
    if deterministic:
        action = torch.argmax(probs)
    else:
        
        action = dist.sample()
    log_prob = dist.log_prob(action)
    return (action.item(), log_prob.reshape(1))

# The agent explores:
# While exploring is provided with rewards
# At the end of the episode we need to evaluate ho good it performed
# This is the goal of the returns.
# RETURN_{0} := gamma^{0} * R_{1} + gamma^{1} * R_{2} + gamma^{2} * R_{3}
# RETURN_{0} := R_{1} + gamma^{1}(R_{2} + gamma^{1}*R_{3})
# RETURN_{0} := R_{1} + gamma^{1}(RETURN_{1})
def compute_returns(rewards, gamma=0.99):
    returns = []
    g_next = 0.0  # Starting from the last move (what will be the next gain? NONE)
    # Consoider all rewards achieved during the game, from the last to the first
    for rew in reversed(rewards):
        # THE GAIN HERE IS WHAT YOU ACHIEVED UNTIL NOW + DISCOUNTEND NEXT REWARDS
        g_next = rew + gamma * g_next
        returns.append(g_next)
    returns.reverse()
    return returns

# This is a personal version of run_episode
# The idea is to keep the same original behavior (defined in utils/legacy)
# but adding the control on observation variables out of bounds defined in evaluate_observations_bounds
def run_episode(env, policy, episode_json_path = None , maxlen=500, scale_factor: float = 1.0, deterministic: bool = False):
    observations = []
    actions = []
    log_probs = []
    rewards = []

    # for each I step I want to evaluate the if the observation_variables went out of bounds
    observations_vars_bounds = []  

    device = next(policy.parameters()).device
    
    (obs, info) = env.reset()
    
    for i in range(maxlen):
        bounds_res = evaluate_observations_bounds(env, (obs,), scale_factor)
        observations_vars_bounds.append(bounds_res)
        # Just as memo, remember that a single (on a step) bound evaluation has this form:
            
        """        {   0: {'is_out_domain': False, 'error_magnitude': 0.0},
                       1: {'is_out_domain': False, 'error_magnitude': 0.0},
                       2: {'is_out_domain': False, 'error_magnitude': 0.0},
                       3: {'is_out_domain': False, 'error_magnitude': 0.0}
                    }
        """
        # one of the element defined above will existf for each step
        # Then:     
        # This is an episode!
        # an episode has several steps
        # I'd like to save as json for each step the bounds aforementioned [see the snippet here before the return]
        # at the end of the episode (in the training):
        #   TAKE THE JSON AND EVALUATE AVERAGE BEHAVIOR OF OBSERVATION VARS

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        (action, log_prob) = select_action(env, obs_tensor, policy, deterministic=deterministic)
        observations.append(obs_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        (obs, reward, term, trunc, info) = env.step(action)

        rewards.append(reward)
        
        if term or trunc:
            final_bounds_res = evaluate_observations_bounds(env, (obs,), scale_factor)
            observations_vars_bounds.append(final_bounds_res)
            break

    # LLM USED HERE: for fastening purposes asked gemini to save evaluation on bounds as json
    if episode_json_path is not None:
        os.makedirs(os.path.dirname(episode_json_path), exist_ok=True)
        payload = {
            "total_steps": len(observations_vars_bounds),
            "bounds_per_step": observations_vars_bounds  
        }
        with open(episode_json_path, "w") as f:
            json.dump(payload, f, indent=4)

    # all the bound will be returned in order to avoid re-opening the json and collect values again to compute
    # average observation space vars behaviors wrt bounds     
    return (observations, actions, torch.cat(log_probs), rewards, observations_vars_bounds)

def get_episode_bounds_statistics(observations_vars_bounds):
    # Within an episode there are several steps,
    # each of them is collected in observations_vars_bounds (which is a list)
    # as:
    """        
    {   0: {'is_out_domain': False, 'error_magnitude': 0.0},
        1: {'is_out_domain': False, 'error_magnitude': 0.0},
        2: {'is_out_domain': False, 'error_magnitude': 0.0},
        3: {'is_out_domain': False, 'error_magnitude': 0.0}
    }
    """
    # the list len is the number of steps
    num_of_steps = len(observations_vars_bounds) 
    if num_of_steps == 0:
        return {}

    # here I'm going to collect the statistics WITHIN the episode, because I want to use them
    # also over the complete training
    stats = {} 

    # Consider all the variables in the observation space 
    num_vars = len(observations_vars_bounds[0]) 

    # ov_idx := "observation variable index"
    # LLM USED HERE: asked for a smart way to iterate over the list of dictionaries, rearranged a bit wrt to original answer
    for ov_idx in range(num_vars):
        # ood := out of domain
        ood_count = sum(1 for step in observations_vars_bounds if step[ov_idx]["is_out_domain"])
        # Actually don't know if "error" is the right word that describe this... sorry I hope it is clear enough
        total_error = sum(step[ov_idx]["error_magnitude"] for step in observations_vars_bounds)
        # for each observation variable consider the percentage of time it went out of domain
        # and, over all the errors consider the mean above all steps
        stats[str(ov_idx)] = {"out_of_domain_ratio": ood_count/num_of_steps, "mean_error_magnitude": total_error/num_of_steps}

    return stats

# Exactly the same of legacy
class PolicyNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, env.action_space.n)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.softmax(self.fc2(s), dim=-1)
        return s
    
BASE_DIR = "experiments"
# NB: Using as base the utils/legacy version of reinforce !!!
# Remember to pass the scale factor for previous defined behaviors


BASE_DIR = "" 

# NB: Using as base the utils/legacy version of reinforce !!!
# Remember to pass the scale factor for previous defined behaviors
def reinforce(output_dir: str, policy, env, val_nn=None, env_render=None, 
              gamma=0.99, num_episodes=10, scale_factor: float = 1.0, 
              n: int = 100, m: int = 10, subtract_sb: bool = False):
    
    experiment_directory = os.path.join(BASE_DIR, output_dir)
    os.makedirs(experiment_directory, exist_ok=True)
    
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    if val_nn is not None:
        optimizer_val_nn = torch.optim.Adam(val_nn.parameters(), lr=1e-2)
    
    running_rewards = [0.0]
    training_bounds_summary = [] 
    
    policy.train()
    if val_nn is not None:
        val_nn.train() 
    
    eval_results = []
    
    for episode in range(num_episodes):
        if episode % n == 0:
            policy.eval() 
            if val_nn is not None:
                val_nn.eval()
            
            validation_tot_reward = 0.0
            validation_tot_length = 0
            
            for i in range(m):
                _, _, _, eval_rewards, _ = run_episode(env, policy, None, scale_factor=scale_factor, deterministic=True)
                validation_tot_reward += sum(eval_rewards)
                validation_tot_length += len(eval_rewards)

            avg_eval_reward = validation_tot_reward / m
            avg_eval_length = validation_tot_length / m
            eval_results.append({"episode": episode, "avg_reward": avg_eval_reward, "avg_length": avg_eval_length})
            print("Validation phase: average reward: {} | average num of steps in episodes: {}".format(avg_eval_reward, avg_eval_length))

            validation_file_path = os.path.join(experiment_directory, "validation_summary.json")
            with open(validation_file_path, "w") as f:
                json.dump(eval_results, f, indent=4)
                
            policy.train()
            if val_nn is not None:
                val_nn.train() 
            
        episode_json_path = os.path.join(experiment_directory, "episode_{}.json".format(episode))
        
        (observations, actions, log_probs, rewards, observations_vars_bounds) = run_episode(
            env, policy, episode_json_path, scale_factor=scale_factor)

        ep_summary = get_episode_bounds_statistics(observations_vars_bounds)

        training_bounds_summary.append({
            "episode": episode, 
            "total_steps": len(observations_vars_bounds),
            "summary": ep_summary
        })

        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float32)
        running_rewards.append(0.05 * returns[0].item() + 0.95 * running_rewards[-1])
        
        if val_nn is not None:
            obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32)
            state_values = val_nn(obs_tensor)
            advantages = returns - state_values.detach()
        else:
            advantages = returns

        if subtract_sb:
            advantages = (advantages - advantages.mean()) / advantages.std()

        opt.zero_grad()
        loss = (-log_probs * advantages).mean()
        loss.backward()
        opt.step()

        if val_nn is not None:
            optimizer_val_nn.zero_grad()
            value_loss = F.mse_loss(state_values, returns)
            value_loss.backward()
            optimizer_val_nn.step()

        if not episode % 100:
            if env_render:
                policy.eval()
                run_episode(env_render, policy, scale_factor=scale_factor) 
                policy.train()
            print(f'Running reward: {running_rewards[-1]}')

    summary_file_path = os.path.join(experiment_directory, "training_summary.json")
    with open(summary_file_path, "w") as f:
        json.dump(training_bounds_summary, f, indent=4)
        
    policy.eval()
    if val_nn is not None:
        val_nn.eval()
    
    return running_rewards, summary_file_path, eval_results


def analyze_global_training_summary(summary_input_path, output_path=None):
    with open(summary_input_path, "r") as f:
        training_summary = json.load(f)
    total_episodes = len(training_summary)
    if total_episodes == 0:
        return {}

    num_vars = len(training_summary[0]["summary"])
    # Once again asked to gemini how to iterate over the json defined above 
    # Smart way to count the number of time the model was out of bound
    episodes_with_oob = sum(
        1 for ep in training_summary
        if any(ep["summary"][str(i)]["out_of_domain_ratio"] > 0 for i in range(num_vars))
    )

    global_analysis = {"total_episodes": total_episodes, "episodes_with_out_of_bounds": episodes_with_oob,
        "ratio_episodes_with_out_of_bounds": episodes_with_oob / total_episodes, "variables": {}}

    for var_idx in range(num_vars):
        var_key = "{}".format(var_idx)
        avg_out_ratio = sum(ep["summary"][var_key]["out_of_domain_ratio"] for ep in training_summary) / total_episodes
        avg_error = sum(ep["summary"][var_key]["mean_error_magnitude"] for ep in training_summary) / total_episodes
        global_analysis["variables"][var_key] = {"mean_out_of_domain_ratio": avg_out_ratio, "mean_error_magnitude": avg_error}

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(global_analysis, f, indent=4)

    return global_analysis


# mode 0 plot only the running rewards and save it
# mode 1 plots running rewards and heatmaps, but for every variable
def plot_running_rewards(running_rewards, mode: int = 1, experiment_dir=None, live_plot:bool = False):
    #
    if experiment_dir is None:
        raise ValueError("Please pass as argument the experiment directory name")
    plots_dir_path = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir_path, exist_ok=True)
    if mode == 0:
        # if mode is 0 I want only one plot for the running rewards
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(running_rewards, label="Running Reward", color='blue', linewidth=2)
        ax.set_title("Training rewards trend")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Running rewards")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plot_path = os.path.join(plots_dir_path, "running_rewards.png")
        plt.savefig(plot_path, dpi=300)
        if live_plot:
            plt.show()

     # else 'Id like to have the plots with the runnings rewards and the out of bound heatmaps for each var in obs space
    elif mode == 1:
        summary_file_path = os.path.join(experiment_dir, "training_summary.json")
        with open(summary_file_path, "r") as f:
            training_summary = json.load(f)
        var_keys = list(training_summary[0]["summary"].keys())
        
        ymin, ymax = min(running_rewards), max(running_rewards)
        num_episodes = len(running_rewards)
        # For each variable we need the values in order to create the heatmap
        for var_key in var_keys:
            oob_per_episode = [ep["summary"][var_key]["out_of_domain_ratio"] * 100 for ep in training_summary]
            fig, ax = plt.subplots(figsize=(10, 5))
            oob_matrix = np.array([oob_per_episode])
            im = ax.imshow(oob_matrix, aspect='auto', cmap='Reds', alpha=0.6, 
                           extent=[0, num_episodes - 1, ymin - 10, ymax + 10], origin='lower')
            
            ax.plot(running_rewards, color='blue', linewidth=2, label='Running Reward')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Var {} Out-of-Domain Ratio (%)'.format(var_key))
            ax.set_xlabel("Episodes")
            ax.set_ylabel("Running Reward")
            ax.set_title("Training Reward con Heatmap (Var {} Out-of-Bounds)".format(var_key))
            ax.legend(loc='lower right')
            plt.tight_layout()
            plot_path = os.path.join(plots_dir_path, f"running_rewards_var_{var_key}.png")
            plt.savefig(plot_path, dpi=300)
            if live_plot:
                plt.show()

import gymnasium as gym  
def execute_experiment(experiment_name:str, env_name:str = "CartPole-v1", 
                       gamma: float = 0.99, num_episodes: int = 500, 
                       scale_factor:float = 1.0, env_render = None,
                       n: int = 100, m: int = 10, live_plot: bool = False, subtract_sb: bool = False, val_nn = None):
    env = gym.make(env_name)
    
    env_render = None 
    policy = PolicyNet(env)
    output_directory = experiment_name
    # For tests I used CARTPOLE WITH SCALING FACTOR OF 0.02 and 0.6
    running_rewards, summary_file_path, eval_results = reinforce(output_dir=output_directory,policy=policy,env=env,env_render=env_render,
        gamma=gamma,num_episodes=num_episodes,scale_factor=scale_factor, n=n, m=m, subtract_sb = subtract_sb, val_nn = val_nn)

    env.close()
    print("Reinforce finished!\n")

    experiment_dir = os.path.dirname(summary_file_path)
    global_summary_file_path = os.path.join(experiment_dir, "global_summary.json")

    # Saving global stats for out of bounds
    # I'd like to plot both the loss with an heatmap behind that represents the 
    # number of outofbounds per episode
    global_stats = analyze_global_training_summary(summary_input_path=summary_file_path, output_path=global_summary_file_path)
    print("> Out of bounds report for the training saved at: {}".format(global_summary_file_path))
    print(f"Total number of out of bounds: {global_stats['episodes_with_out_of_bounds']}/{global_stats['total_episodes']} ({global_stats['ratio_episodes_with_out_of_bounds']*100:.1f}%)")
    for var_name, stats in global_stats['variables'].items():
        print("Observation Space variable: {} -> Average perc out of bounds:" \
        "{} with Average error: {}".format(var_name, stats['mean_out_of_domain_ratio'], stats['mean_error_magnitude']))

    plot_running_rewards(running_rewards, mode = 1, experiment_dir=experiment_dir, live_plot=live_plot) 
    plot_validation_rewards(eval_results, experiment_dir=experiment_dir, live_plot=live_plot)
    print("Experiment is finished. All produced content can be found at: {}".format(experiment_dir))
    return experiment_dir

# This was added after for plotting and saving eval results:
def plot_validation_rewards(eval_results, experiment_dir, live_plot:bool = False):
    if experiment_dir is None:
        raise ValueError("Please pass as argument the experiment directory name")
    plots_dir_path = os.path.join(experiment_dir, "plots")
    os.makedirs(plots_dir_path, exist_ok=True)

    episodes = [res["episode"] for res in eval_results]
    avg_rewards = [res["avg_reward"] for res in eval_results]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, avg_rewards, label="Validation Rewards", color='orange')
    ax.set_title("Validation rewards")
    ax.set_xlabel("Episods")
    ax.set_ylabel("Average Reward (over M episodes)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(plots_dir_path, "validation_rewards.png")
    plt.savefig(plot_path, dpi=300)
    if live_plot:
        plt.show()


# Adding a Value State NN
# "Typically we same the same network architecture as that of policy"
class ValueNet(nn.Module):
    def __init__(self, environment):
        super().__init__()
        self.fc1 = nn.Linear(environment.observation_space.shape[0], 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = self.fc2(s) 
        return s.squeeze(-1)

## A2C


def dummy_a2c(env, actor, critic, opt_actor, opt_critic, gamma=0.99, maxlen=500, seed=None, scale_factor=1.0, episode_json_path=None):
    device = next(actor.parameters()).device
    total_reward = 0.0
    observations_vars_bounds = []
    
    # This is done in order to start the env always with the same settings
    if seed is not None:
        (obs, info) = env.reset(seed=seed)
    else:
        (obs, info) = env.reset()

    opt_actor.zero_grad()
    opt_critic.zero_grad()

    # for each step
    for i in range(maxlen):

        bounds_res = evaluate_observations_bounds(env, (obs,), scale_factor)
        observations_vars_bounds.append(bounds_res)

        # consider observations
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # select an actio via the policy (actor)
        (action, log_prob) = select_action(env, obs_tensor, actor)

        # When the agent makes a step it reaches another state
        #  This means that we observes new realizations of the observation space variables
        #  Also the agent receives a reward from the env
        (next_obs, reward, term, trunc, info) = env.step(action)
        done = term or trunc
        total_reward += reward
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # Critic's role is to evaluate if the chosen action brought an ADVANTAGE or NOT
        #  How it does it?
        #    Please consider the state from which the agent started and compute state value
        v_s = critic(obs_tensor)
        
        #    The agent is arriving in a new state, can you estimate also that state value?
        #    
        with torch.no_grad():
            v_s_next = critic(next_obs_tensor) if not done else torch.tensor(0.0, device=device)

        # delta <-- R + \gamma * \hat{v}(s^{'}, w) - \hat{v}(s,w)
        delta = reward + gamma * v_s_next - v_s

        # computing losses
        critic_loss = delta.pow(2)
        actor_loss = -log_prob * delta.detach()

        # updating losses
        actor_loss.backward()
        critic_loss.backward()
        
        obs = next_obs
        if done:
            final_bounds_res = evaluate_observations_bounds(env, (obs,), scale_factor)
            observations_vars_bounds.append(final_bounds_res)
            break

    opt_actor.step()
    opt_critic.step()

    if episode_json_path is not None:
        os.makedirs(os.path.dirname(episode_json_path), exist_ok=True)
        payload = {
            "total_steps": len(observations_vars_bounds),
            "bounds_per_step": observations_vars_bounds,
        }
        with open(episode_json_path, "w") as f:
            json.dump(payload, f, indent=4)
    
    return total_reward, observations_vars_bounds

def a2c_train(output_dir: str, actor, critic, env, seed=None, num_episodes=1000, gamma=0.99, n=50, m=10, scale_factor=1.0):
    experiment_directory = output_dir
    os.makedirs(experiment_directory, exist_ok=True)

    opt_actor = optim.Adam(actor.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
    running_rewards = [0.0]
    training_history = []
    eval_results = []
    training_bounds_summary = []

    for episode in range(num_episodes):
        actor.train()
        critic.train()

        current_seed = seed if episode == 0 else None
        episode_json_path = os.path.join(experiment_directory, "episode_{}.json".format(episode))

        episode_reward, obs_bounds = dummy_a2c(env,actor,critic,opt_actor,opt_critic,
                                               gamma=gamma,maxlen=500,seed=current_seed,
                                               scale_factor=scale_factor,episode_json_path=episode_json_path)
        ep_summary = get_episode_bounds_statistics(obs_bounds)
        training_bounds_summary.append(
            {
                "episode": episode,
                "total_steps": len(obs_bounds),
                "summary": ep_summary,
            }
        )

        if len(running_rewards) == 1 and running_rewards[0] == 0.0:
            running_rewards[0] = episode_reward
        else:
            running_rewards.append(0.05 * episode_reward + 0.95 * running_rewards[-1])

        training_history.append({"episode": episode, "reward": episode_reward,"running_reward": running_rewards[-1]})

        if episode % n == 0:
            actor.eval()
            critic.eval()
            val_tot_reward = 0.0
            for _ in range(m):
                _, _, _, eval_rewards, _ = run_episode(env, actor, None, scale_factor=scale_factor, deterministic=True)
                val_tot_reward += sum(eval_rewards)
            avg_eval_reward = val_tot_reward / m
            eval_results.append({"episode": episode, "avg_reward": avg_eval_reward})
            print("EPISODE: {}".format(episode))
            print("TRAIN: Ep Reward: {} | Running Reward: {}".format(episode_reward, running_rewards[-1]))
            print("Evaluation avg reward: {}".format(avg_eval_reward))

    
    summary_file_path = os.path.join(experiment_directory, "training_summary.json")
    with open(summary_file_path, "w") as f:
        json.dump(training_bounds_summary, f, indent=4)

    eval_file = os.path.join(experiment_directory, "validation_summary.json")
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    global_summary_file_path = os.path.join(experiment_directory, "global_summary.json")

    global_stats = analyze_global_training_summary(summary_input_path=summary_file_path, output_path=global_summary_file_path)
    print("> Out of bounds report at: {}".format(global_summary_file_path))

    actor.eval()
    critic.eval()
    return running_rewards, eval_results

####

class PolicyNetDeeper(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)  
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        s = F.softmax(self.fc4(s), dim=-1)
        return s

class ValueNetDeeper(nn.Module):
    def __init__(self, environment):
        super().__init__()
        input_dim = environment.observation_space.shape[0]
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1) 
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        v = self.fc4(s)
        return v.squeeze(-1)

