import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

def preprocess(img):
    """
    Preprocess CarRacing-v3 image:
    - Crop specific region
    - Resize to 84x84
    - Convert to grayscale
    - Normalize to [0, 1]
    """
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img.astype(np.float32)

class CarRacingEnv(gym.Wrapper):
    def __init__(
        self,
        env_id="CarRacing-v3",
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        render_mode="rgb_array",
        **kwargs
    ):
        env = gym.make(env_id, render_mode=render_mode, continuous=False, **kwargs)
        super(CarRacingEnv, self).__init__(env)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, 84, 84),
            dtype=np.float32,
        )

    def reset(self, seed=0, options=None):
        """
        Resets environment with initial no-op steps and returns stacked frames.
        """
        s, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.initial_no_op):
            s, _, terminated, truncated, _ = self.env.step(0)
            if terminated or truncated:
                s, info = self.env.reset(seed=seed, options=options)

        s = preprocess(s)
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))
        return self.stacked_state, info

    def step(self, action):
        """
        Steps through the environment with action repeat and returns stacked observation.
        """
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break

        reward = np.clip(reward, -1.0, 1.0)
        s = preprocess(s)
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        return self.stacked_state, reward, terminated, truncated, info

    def render(self):
        """
        Return the most recent rendered frame from the base environment.
                    
        Returns:
            np.ndarray: RGB image from the base environment.
        """
        img = self.env.render()
        return img

    def close(self):
        """Close the underlying environment to release memory."""
        self.env.close()
