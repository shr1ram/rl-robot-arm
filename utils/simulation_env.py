import numpy as np
import mujoco
from mujoco import viewer


class SimulationEnv:
    def __init__(self, xml_file_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)

        self.max_height = self._get_max_lift_height()
        self.initial_cube_z = self._get_cube_z_position()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()
        return obs, reward, done, {}

    def reset(self):
        self.data = mujoco.MjData(self.model)
        return self._get_obs()

    def _get_obs(self):
        # Customize based on the observations needed
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
        ])

    def _compute_reward(self):
        cube_z = self._get_cube_z_position()
        reward = cube_z - self.initial_cube_z
        return reward

    def _check_done(self):
        cube_z = self._get_cube_z_position()
        return cube_z >= self.max_height

    def _get_cube_z_position(self):
        # Adjust the site or geom name based on model
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        return self.data.geom_xpos[cube_geom_id][2]

    def _get_max_lift_height(self):
        # Placeholder: you can return a constant or define based on the model
        return 0.2  # e.g., max target height

    def render(self):
        if not hasattr(self, '_viewer'):
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()