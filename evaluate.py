from stable_baselines3 import PPO
import time

from env import SearchEnv

env = SearchEnv(training=False, cfg="./base_config.yaml")
ob, _ = env.reset()

model = PPO.load("ppo_search", env=env)

for i in range(10000):
    action, _states = model.predict(ob)
    #print(action)
    ob, rewards, terms, truncs, info = env.step(action)
    if terms or truncs: break