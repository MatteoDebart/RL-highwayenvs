import gymnasium as gym
from roundabout_reward_wrapper import RoundaboutRewardWrapper
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


def make_eval_env():
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    env.unwrapped.configure({"duration": 20})
    env = RoundaboutRewardWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load("stablebaselines_roundabout/roundabout_checkpoints/old_version/vecnormalize_stats.pkl", env)
    #env = gym.wrappers.RecordVideo(env, video_folder="./vid", episode_trigger=lambda x: True)
    env.training = False  # disable normalization updates during eval
    env.norm_reward = False
    return env


# --- Random Agent Implementation ---
class RandomAgent:
    def __init__(self, env:gym.Env):
        self.env=env
    
    def predict(self, observation, **kwargs):
        act=self.env.action_space.sample()
        obs = self.env.observation_space.sample()
        return [act], [obs]
