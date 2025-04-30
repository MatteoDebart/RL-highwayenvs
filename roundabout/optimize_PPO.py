import gymnasium as gym
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from roundabout_reward_wrapper import RoundaboutRewardWrapper

def optimize_ppo(trial):
    # Sample hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 1.0)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.4)
    ent_coef = trial.suggest_loguniform('ent_coef', 1e-6, 0.01)

    # Make and normalize environment
    env = gym.make("roundabout-v0")
    env.unwrapped.configure({"duration": 20})

    wrapped_env = RoundaboutRewardWrapper(env)
    wrapped_env = DummyVecEnv([lambda: wrapped_env])
    wrapped_env = VecNormalize(wrapped_env, norm_obs=True, norm_reward=True)

    # Initialize PPO model with the sampled hyperparameters
    model = PPO("MlpPolicy",
                wrapped_env,
                learning_rate=learning_rate,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=1)

    # Train the model for a smaller number of timesteps
    model.learn(total_timesteps=10_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    return mean_reward

# --- Create and Save Optuna Study ---
study_storage = "roundabout_checkpoints/optuna_study.db"  # SQLite file to store results
study = optuna.create_study(direction='maximize', storage=f"sqlite:///{study_storage}", load_if_exists=True)
study.optimize(optimize_ppo, n_trials=20)

# Print the best trial parameters
best_trial = study.best_trial
best_params = best_trial.params
print("Best Trial Parameters:", best_params)

# --- Retrain the Model Using Best Parameters ---
# Use the best hyperparameters to retrain the model for a longer period
env = gym.make("roundabout-v0")
env.unwrapped.configure({"duration": 20})

wrapped_env = RoundaboutRewardWrapper(env)
wrapped_env = DummyVecEnv([lambda: wrapped_env])
wrapped_env = VecNormalize(wrapped_env, norm_obs=True, norm_reward=True)

# Initialize the model with the best parameters
model = PPO("MlpPolicy", wrapped_env,
            learning_rate=best_params['learning_rate'],
            gamma=best_params['gamma'],
            gae_lambda=best_params['gae_lambda'],
            clip_range=best_params['clip_range'],
            ent_coef=best_params['ent_coef'],
            verbose=1)

# Retrain the model for a larger number of timesteps
model.learn(total_timesteps=10_000)

# Save the final trained model
model.save("roundabout_checkpoints_optuna")
print("Final PPO model saved.")

