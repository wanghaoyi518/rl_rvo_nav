import gym
from gym_env.envs.ir_gym import ir_gym

class mrnav(gym.Env):
    def __init__(self, world_name=None, neighbors_region=5, neighbors_num=10, **kwargs):
        
        self.ir_gym = ir_gym(world_name, neighbors_region, neighbors_num, **kwargs)
        
        self.observation_space = self.ir_gym.observation_space
        self.action_space = self.ir_gym.action_space
        
        self.neighbors_region = neighbors_region
        self.circular = kwargs.get('circle', [5, 5, 4])
        self.square = kwargs.get('square', [0, 0, 10, 10])
        self.interval = kwargs.get('interval', 1)

    def step(self, action, vel_type='diff', stop=True, **kwargs):
        # Use deadlock resolution if enabled
        if hasattr(self.ir_gym, 'enable_deadlock_resolution') and self.ir_gym.enable_deadlock_resolution:
            return self.ir_gym._step_with_deadlock_resolution(action)
        else:
            # Original step logic
            return self.step_ir(action, vel_type=vel_type, stop=stop, **kwargs)
    
    # def step_original(self, action, vel_type='diff', stop=True, **kwargs):
    def step_ir(self, action, vel_type='diff', stop=True, **kwargs):

        if not isinstance(action, list):
            action = [action]

        rvo_reward_list = self.ir_gym.rvo_reward_list_cal(action)
        self.ir_gym.robot_step(action, vel_type=vel_type, stop=stop)
        self.ir_gym.obs_cirs_step()
        obs_list, mov_reward, done_list, info_list = self.ir_gym.obs_move_reward_list(action, **kwargs)

        reward_list = [x+y for x, y in zip(rvo_reward_list, mov_reward)]
        
        return obs_list, reward_list, done_list, info_list

    def render(self, mode = 'human', save=False, path=None, i = 0, **kwargs):
        self.ir_gym.render(0.01, **kwargs)

        if save:
            self.ir_gym.save_fig(path, i) 

    def reset(self, mode=0, **kwargs):
        # mode = kwargs.get('mode', 0)
        return self.ir_gym.env_reset(mode, circular = self.circular, square=self.square, interval=self.interval, **kwargs)

    def reset_one(self, id):
        self.ir_gym.components['robots'].robot_reset(id)

    def show(self):
        self.ir_gym.show()
    
    # Deadlock resolution mode control methods
    def enable_deadlock_resolution_mode(self, config_file=None):
        """Enable deadlock resolution mode for testing."""
        self.ir_gym.enable_deadlock_resolution_mode(config_file)
    
    def disable_deadlock_resolution_mode(self):
        """Disable deadlock resolution mode (restore pure RL)."""
        self.ir_gym.disable_deadlock_resolution_mode()
    
    def is_in_deadlock_resolution_mode(self):
        """Check if in deadlock resolution mode."""
        return self.ir_gym.is_in_deadlock_resolution_mode()
    
    def get_current_mode(self, agent_id):
        """Get current mode of an agent."""
        return self.ir_gym.get_current_mode(agent_id)