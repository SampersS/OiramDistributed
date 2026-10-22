import gymnasium as gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import A2C

env = gym.make("SuperMarioBros-v0", render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    actions, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(actions)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
