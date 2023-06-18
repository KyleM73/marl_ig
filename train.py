import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import torch

from env import SearchEnv
from models import MlpExtractor, CnnExtractor

register(
    id="SearchEnv-v1",
    # path to the class for creating the env
    # Note: entry_point also accept a class as input (and not only a string)
    entry_point=SearchEnv,
    kwargs={
        "training" : True,
        "cfg"      : "./base_config.yaml",
    }
)

def makeEnvs(env_id):
    def _init():
        env = gym.make(env_id)
        return env
    return _init

if __name__ == "__main__":
    env_id = "SearchEnv-v1"
    num_envs = 8
    n_steps = 32
    buffer_size = n_steps * num_envs
    batch_size = 256
    assert buffer_size % batch_size == 0

    train_steps = 500_000
    device = torch.device("mps")

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,features_extractor_class=CnnExtractor,
        features_extractor_kwargs=dict(data_dim=256,context_dim=16,activation_fn=torch.nn.ReLU),
        net_arch=[256,256,16],normalize_images=False)

    vec_env = SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)])
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs,
        n_steps=n_steps, batch_size=batch_size, seed=vec_env.get_attr("cfg")[0]["SEED"],
        verbose=1, device=device) #if continuous action space: use_sde=True
    model.learn(total_timesteps=train_steps,progress_bar=True)
    model.save("ppo_search")
