import gymnasium  as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback



class RoundaboutRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.entered_roundabout = False

    def reset(self, **kwargs):
        self.entered_roundabout = False
        return self.env.reset(**kwargs)

    def reward(self, reward):
        additional_reward = 0

        vehicle = self.env.unwrapped.vehicle
        lane_index = vehicle.lane_index
        road = self.env.unwrapped.road
        lane = road.network.get_lane(lane_index)

        # Detect if entered roundabout
        if isinstance(lane, highway_env.road.lane.CircularLane):
            self.entered_roundabout = True

        # Reward exiting the roundabout after entering
        if isinstance(lane, highway_env.road.lane.StraightLane) and self.entered_roundabout:
            self.entered_roundabout = False
            additional_reward += 5

        # Penalize going off road
        if not vehicle.on_road:
            additional_reward += 10

        return reward + additional_reward

class CustomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.env.unwrapped.vehicle.on_road:
            terminated = True
        return obs, reward, terminated, truncated, info


env = gym.make("roundabout-v0")
env.unwrapped.configure({
    "duration": 20})
wrapped_env = RoundaboutRewardWrapper(env)
wrapped_env = CustomWrapper(wrapped_env)

checkpoint_callback = CheckpointCallback(
    save_freq=5_000,
    save_path="./checkpoints/",
    name_prefix="ppo_roundabout"
)

# Initialize the PPO model with a policy and the environment
model = PPO("MlpPolicy", wrapped_env, verbose=1, learning_rate=1e-2, n_epochs=30, batch_size=32)

# Train the agent for 10,000 time steps

model.learn(total_timesteps=20000, progress_bar=True, callback = checkpoint_callback )

model.save("roundabout_checkpoints/ppo_roundabout_custom_reward")
