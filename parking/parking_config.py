import gymnasium as gym
import highway_env
from matplotlib import pyplot as plt
import pickle

config = {
    "observation": {
        "type": "KinematicsGoal",
        "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
        "scales": [100, 100, 5, 5, 1, 1],
        "normalize": False,
    },
    "action": {"type": "ContinuousAction"},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 300,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False,
    "manual_control": False,
    "real_time_rendering": False,
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
    "success_goal_reward": 0.12,
    "collision_reward": -5,
    "steering_range": 0.7853981633974483,
    "duration": 50,
    "controlled_vehicles": 1,
    "vehicles_count":12,
    "add_walls": True,
}
with open("config_task2.pkl", "wb") as f:
    pickle.dump(config, f)