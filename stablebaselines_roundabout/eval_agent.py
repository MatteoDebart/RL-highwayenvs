from stable_baselines3 import PPO
from utils import make_eval_env

def display_runs(n_episodes):
    # Create the evaluation environment
    eval_env = make_eval_env()

    # --- Load model ---
    model = PPO.load("stablebaselines_roundabout/roundabout_checkpoints/ppo_roundabout_custom_reward")

    # --- Evaluation ---
    episode_rewards = []
    print("Starting evaluation...")
    # --- Evaluate episodes ---
    for ep in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        episode_steps = []  # To store the steps for the current episode

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            done = done or not eval_env.envs[0].unwrapped.vehicle.on_road  # Ensure the vehicle is on the road

            total_reward += reward[0]
            episode_steps.append((obs, action, reward))  # Save state, action, reward
            eval_env.render()
        # Store the total reward and steps for the current episode
        episode_rewards.append(total_reward)
        print(f"Episode {ep}: Reward {total_reward}")

    #eval_env.close_video_recorder()
    eval_env.close()

if __name__=='__main__':
    n_episodes=10
    display_runs(n_episodes)