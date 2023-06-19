import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import torch

from callbacks import ImageRecorderCallback
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
    num_envs = 4
    n_steps = 300
    buffer_size = n_steps * num_envs
    batch_size = 1200
    assert buffer_size % batch_size == 0
    log_path = "./logs/6_19_23"

    train_steps = 10_000_000
    device = torch.device("cuda")

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,features_extractor_class=CnnExtractor,
        features_extractor_kwargs=dict(data_dim=256,context_dim=16,activation_fn=torch.nn.ReLU),
        net_arch=[256,256,256],normalize_images=False)

    vec_env = VecMonitor(SubprocVecEnv([makeEnvs(env_id) for i in range(num_envs)], start_method="forkserver")) #spawn, fork, forkserver (fork is not threadsafe)
    #vec_env = DummyVecEnv([makeEnvs(env_id) for i in range(num_envs)])
    eval_env = VecMonitor(SubprocVecEnv([makeEnvs(env_id) for i in range(1)], start_method="forkserver"))
    img_callbak = ImageRecorderCallback()
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path,
        eval_freq=max(100000//num_envs,1), deterministic=True, render=False, verbose=1,
        callback_after_eval=img_callbak)
    model = PPO("MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs,
        n_steps=n_steps, batch_size=batch_size, seed=vec_env.get_attr("cfg")[0]["SEED"],
        tensorboard_log=log_path, verbose=1, device=device) #if continuous action space: use_sde=True
    model.learn(total_timesteps=train_steps, callback=eval_callback, progress_bar=True)
    model.save("{}/ppo_search_final".format(log_path))
