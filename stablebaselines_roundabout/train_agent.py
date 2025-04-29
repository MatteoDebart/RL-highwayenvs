import gymnasium as gym
import optuna
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from roundabout_reward_wrapper import RoundaboutRewardWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import csv

# best param
study_storage = "stablebaselines_roundabout/roundabout_checkpoints/optuna_study.db"

study = optuna.load_study(study_name="no-name-035cbc85-e68f-446a-b5bd-e7cf79f0936c",
                          storage=f"sqlite:///{study_storage}")
best_trial = study.best_trial
best_params = best_trial.params
print("Best Hyperparameters from Optuna Study:", best_params)

# --- Callback to track the rewards --- 
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals.get('rewards', [0])[0]
        self.rewards.append(reward)
        return True
    
    def get_rewards(self):
        return self.rewards

# --- Retrain Model Using Best Hyperparameters ---
env = gym.make("roundabout-v0")
env.unwrapped.configure({"duration": 20})
wrapped_env = RoundaboutRewardWrapper(env)
wrapped_env = DummyVecEnv([lambda: wrapped_env])
wrapped_env = VecNormalize(wrapped_env, norm_obs=True, norm_reward=True)

# Model à entrainer
model = PPO("MlpPolicy", wrapped_env, verbose=1)
""", 
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            gae_lambda=best_params['gae_lambda'],
            clip_range=best_params['clip_range'],
            ent_coef=best_params['ent_coef'])"""

reward_callback = RewardCallback()
model.learn(total_timesteps=20_000, callback=reward_callback)
rewards = reward_callback.get_rewards()

# save the rewards
with open('rewards.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Reward'])  # Write header
    for reward in rewards:
        writer.writerow([reward])  # Write each reward in a new row

# --- Plot the Results ---
# 1. Plotting the Reward Distribution
plt.figure(figsize=(10, 5))
plt.hist(rewards, bins=30, color='skyblue', edgecolor='black')
plt.title("Reward Distribution")
plt.xlabel("Reward")
plt.ylabel("Frequency")
plt.show()

# 2. Plotting the Rewards over Training Steps
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title("Rewards Over Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Reward")
plt.show()

# Save the trained model and environment stats
model.save("stablebaselines_roundabout/roundabout_checkpoints/ppo_roundabout_custom_reward")
wrapped_env.save("stablebaselines_roundabout/roundabout_checkpoints/vecnormalize_stats.pkl")
