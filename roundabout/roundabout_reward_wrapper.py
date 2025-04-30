import gymnasium  as gym
import highway_env

class RoundaboutRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env, )
        self.entered_roundabout = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.env.unwrapped.vehicle.on_road:
            terminated = True
        return obs, reward, terminated, truncated, info
    
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

        # Reward going all the way
        if not vehicle.on_road:
            additional_reward += 10

        return reward + additional_reward
