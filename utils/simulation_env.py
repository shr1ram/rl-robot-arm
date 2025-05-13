import numpy as np
import mujoco
from mujoco import viewer


class SimulationEnv:
    def __init__(self, xml_file_path: str, max_steps=None):
        self.model = mujoco.MjModel.from_xml_path(xml_file_path)
        self.data = mujoco.MjData(self.model)

        self.max_height = self._get_max_lift_height()
        self.initial_cube_z = self._get_cube_z_position()
        
        # Initialize previous positions for tracking movement
        self.prev_left_pos = None
        self.prev_right_pos = None
        
        # Step counter for episode length limiting
        self.step_counter = 0
        self.max_steps = max_steps  # None means no step limit

    def step(self, action):
        # Apply action and step the simulation
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        # Increment step counter
        self.step_counter += 1
        
        # Calculate reward
        obs = self._get_obs()
        reward = self._compute_reward()
        
        # Check if episode should terminate due to cube height
        done = self._check_done()
        if self.step_counter % 100 == 0:  # Print with 10% chance to avoid too much output
            print(self.step_counter)
        # Also check if episode should terminate due to step limit
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True
            print(f"Episode terminated due to reaching max steps: {self.max_steps}")
        
        return obs, reward, done, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # Reset step counter when environment is reset
        self.step_counter = 0
        # Reset previous positions
        self.prev_left_pos = None
        self.prev_right_pos = None
        return self._get_obs()

    def _get_obs(self):
        # Customize based on the observations needed
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
        ])

    def _compute_reward(self):
        # Get cube position
        cube_pos = self._get_cube_position()
        cube_z = cube_pos[2]
        height_reward = cube_z - self.initial_cube_z
        
        # Calculate surface-to-surface distance between chopsticks and cube
        left_surface_dist = self._get_surface_distance("left_stick", "cube")
        right_surface_dist = self._get_surface_distance("right_stick", "cube")
        
        # Calculate individual chopstick distance rewards
        left_distance_reward = -left_surface_dist
        right_distance_reward = -right_surface_dist
        
        # Calculate alignment reward (penalize deviation from x=0, y=0)
        # Use the horizontal distance (x,y) from the origin
        horizontal_deviation = np.sqrt(cube_pos[0]**2 + cube_pos[1]**2)
        alignment_reward = -horizontal_deviation  # Negative because we want to minimize deviation
        
        # Combine rewards with balanced weights
        reward = (10.0 * height_reward + 
                 200.0 * left_distance_reward + 
                 200.0 * right_distance_reward + 
                 8.0 * alignment_reward)
        
        # Print debug info occasionally
        if self.step_counter % 100 == 0:  # Print in ~1% of steps
            print(f"DEBUG - Height: {height_reward:.4f}, Left Dist: {left_surface_dist:.4f}, "
                  f"Right Dist: {right_surface_dist:.4f}, Alignment: {horizontal_deviation:.4f}, "
                  f"Reward: {reward:.4f}")
        
        return reward

    def _check_done(self):
        # Check if cube has reached the target height
        cube_z = self._get_cube_z_position()
        height_reached = cube_z >= self.max_height
        
        if height_reached and np.random.random() < 0.1:  # Print with 10% chance
            print("Simulation complete - Maximum height reached!")
            
        return height_reached

    def _get_cube_z_position(self):
        # Adjust the site or geom name based on model
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        return self.data.geom_xpos[cube_geom_id][2]
        
    def _get_cube_position(self):
        # Get full 3D position of the cube
        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        return self.data.geom_xpos[cube_geom_id]
        
    def _get_chopstick_position(self, stick_name):
        # Get the position of a chopstick
        stick_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, stick_name)
        return self.data.xpos[stick_body_id]
        
    def _get_surface_distance(self, obj1_name, obj2_name):
        """Calculate the surface-to-surface distance between two objects
        
        This uses MuJoCo's built-in collision detection to find the minimum distance
        between the surfaces of two objects.
        """
        # Get geom IDs for the objects
        if obj1_name == "left_stick" or obj1_name == "right_stick":
            # For chopsticks, we need to get the geom associated with the body
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj1_name)
            # Get all geoms associated with this body
            geom1_id = None
            for i in range(self.model.nbody):
                if i == body_id:
                    # Find the first geom associated with this body
                    for j in range(self.model.ngeom):
                        if self.model.geom_bodyid[j] == body_id:
                            geom1_id = j
                            break
                    break
        else:
            # For other objects like the cube, we can get the geom directly
            geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj1_name)
            
        # Get the second geom ID
        if obj2_name == "left_stick" or obj2_name == "right_stick":
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj2_name)
            geom2_id = None
            for i in range(self.model.nbody):
                if i == body_id:
                    for j in range(self.model.ngeom):
                        if self.model.geom_bodyid[j] == body_id:
                            geom2_id = j
                            break
                    break
        else:
            geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj2_name)
        
        # If we couldn't find the geoms, fall back to center-to-center distance
        if geom1_id is None or geom2_id is None:
            pos1 = self._get_object_position(obj1_name)
            pos2 = self._get_object_position(obj2_name)
            return np.linalg.norm(pos1 - pos2)
        
        # Get positions and sizes of the geoms
        pos1 = self.data.geom_xpos[geom1_id]
        pos2 = self.data.geom_xpos[geom2_id]
        
        # Calculate center-to-center distance
        center_dist = np.linalg.norm(pos1 - pos2)
        
        # Get the sizes of the geoms
        # For capsules (chopsticks), size[0] is radius, size[1] is half-length
        # For boxes (cube), size is half-width in each dimension
        geom1_type = self.model.geom_type[geom1_id]
        geom2_type = self.model.geom_type[geom2_id]
        
        # Get the appropriate sizes based on geom types
        if geom1_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            size1 = self.model.geom_size[geom1_id][0]  # Radius of capsule
        elif geom1_type == mujoco.mjtGeom.mjGEOM_BOX:
            # For a box, use the minimum half-width as an approximation
            size1 = min(self.model.geom_size[geom1_id])
        else:
            # Default to a small value for unknown geom types
            size1 = 0.01
            
        if geom2_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            size2 = self.model.geom_size[geom2_id][0]  # Radius of capsule
        elif geom2_type == mujoco.mjtGeom.mjGEOM_BOX:
            # For a box, use the minimum half-width as an approximation
            size2 = min(self.model.geom_size[geom2_id])
        else:
            # Default to a small value for unknown geom types
            size2 = 0.01
        
        # Surface distance = center distance - sum of radii
        # Clamp to zero to avoid negative distances when objects overlap
        surface_dist = max(0.0, center_dist - size1 - size2)
        
        return surface_dist
        
    def _get_object_position(self, obj_name):
        """Get the position of any object (body or geom)"""
        if obj_name == "left_stick" or obj_name == "right_stick":
            return self._get_chopstick_position(obj_name)
        else:
            return self._get_cube_position()

    def _get_max_lift_height(self):
        # Placeholder: you can return a constant or define based on the model
        return 0.6  # e.g., max target height

    def render(self):
        if not hasattr(self, '_viewer'):
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()