import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        best_idx = np.argmax(self.training_env.get_attr("episode_rewards"))
        image = self.training_env.get_attr("entropy")[best_idx]
        image = image.reshape(image.shape[0], image.shape[1], 1)
        image *= 0.5
        image += 0.5
        image_baseline = self.training_env.get_attr("map")[best_idx]
        image_baseline = image_baseline.reshape(image_baseline.shape[0], image_baseline.shape[1], 1)
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("entropy/entropy", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        self.logger.record("entropy/map", Image(image_baseline, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True