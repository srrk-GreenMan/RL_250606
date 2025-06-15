import numpy as np
import ray

from car_racing_env import CarRacingEnv


@ray.remote
class EnvWorker:
    def __init__(self):
        self.env = CarRacingEnv()

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class RayVectorEnv:
    """Simple vectorized environment using Ray actors."""

    def __init__(self, num_envs):
        ray.init(ignore_reinit_error=True)
        self.num_envs = num_envs
        self.workers = [EnvWorker.remote() for _ in range(num_envs)]

        temp_env = CarRacingEnv()
        self.single_observation_space = temp_env.observation_space
        self.single_action_space = temp_env.action_space
        temp_env.close()

    def reset(self, seed=None):
        seeds = [None] * self.num_envs if seed is None else [seed + i for i in range(self.num_envs)]
        results = ray.get([w.reset.remote(seed=s) for w, s in zip(self.workers, seeds)])
        states, infos = zip(*results)
        return np.stack(states), list(infos)

    def step(self, actions):
        results = ray.get([w.step.remote(a) for w, a in zip(self.workers, actions)])
        states, rewards, terminateds, truncateds, infos = zip(*results)
        return (
            np.stack(states),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=np.float32),
            np.array(truncateds, dtype=np.float32),
            list(infos),
        )

    def close(self):
        for w in self.workers:
            w.close.remote()
        ray.shutdown()
