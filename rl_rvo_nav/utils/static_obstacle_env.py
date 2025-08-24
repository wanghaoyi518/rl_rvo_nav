#!/usr/bin/env python3
"""
Random static obstacle environment wrapper
"""

import os
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from rl_rvo_nav.utils.static_obstacle_generator import StaticObstacleGenerator

class StaticObstacleEnvWrapper:
    """Wrapper for gym environment to support random static obstacles"""
    
    def __init__(self,
                 base_world_config_path: str,
                 obstacle_count: int = 4,
                 obstacle_size: float = 1.0,
                 min_obstacle_distance: float = 2.0,
                 world_size: Tuple[float, float] = (10.0, 10.0),
                 temp_dir: Optional[str] = None):
        """
        Initialize random static obstacle environment wrapper
        
        Args:
            base_world_config_path: Path to base world configuration
            obstacle_count: Number of obstacles per episode
            obstacle_size: Size of square obstacles
            min_obstacle_distance: Minimum distance between obstacles
            world_size: World dimensions (width, height)
            temp_dir: Directory for temporary config files
        """
        self.base_world_config_path = base_world_config_path
        self.temp_dir = temp_dir or "."
        
        # Load base configuration
        with open(base_world_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
            
        # Initialize obstacle generator
        self.obstacle_generator = StaticObstacleGenerator(
            world_width=world_size[0],
            world_height=world_size[1],
            obstacle_size=obstacle_size,
            obstacle_count=obstacle_count,
            min_obstacle_distance=min_obstacle_distance
        )
        
        # Temporary config file path
        self.temp_config_path = None
        self.episode_count = 0
        
    def generate_episode_config(self, 
                              robot_positions: Optional[List[Tuple[float, float]]] = None,
                              robot_goals: Optional[List[Tuple[float, float]]] = None) -> str:
        """
        Generate a new world configuration with random obstacles for this episode
        
        Args:
            robot_positions: Robot initial positions
            robot_goals: Robot goal positions
            
        Returns:
            Path to temporary configuration file
        """
        self.episode_count += 1
        
        # Generate new obstacle layout
        config = self.obstacle_generator.create_world_config(
            self.base_config, robot_positions, robot_goals
        )
        
        # Create temporary file
        temp_filename = f"static_world_episode_{self.episode_count}.yaml"
        self.temp_config_path = os.path.join(self.temp_dir, temp_filename)
        
        # Save to temporary file
        self.obstacle_generator.save_config_to_file(config, self.temp_config_path)
        
        return self.temp_config_path
    
    def cleanup_temp_files(self):
        """Clean up temporary configuration files"""
        if self.temp_config_path and os.path.exists(self.temp_config_path):
            try:
                os.remove(self.temp_config_path)
            except OSError:
                pass
                
    def get_current_config_path(self) -> str:
        """Get current temporary config path"""
        return self.temp_config_path or self.base_world_config_path
        
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup_temp_files()


class StaticObstacleGymWrapper:
    """Gym environment wrapper with random static obstacles"""
    
    def __init__(self, env_class, static_env_wrapper: StaticObstacleEnvWrapper, **env_kwargs):
        """
        Initialize gym wrapper
        
        Args:
            env_class: Original gym environment class
            static_env_wrapper: Random static obstacle environment wrapper
            **env_kwargs: Environment creation arguments
        """
        self.env_class = env_class
        self.static_wrapper = static_env_wrapper
        self.env_kwargs = env_kwargs
        self.env = None
        self._reset_count = 0
        
    def _create_env_with_obstacles(self) -> Any:
        """Create environment with current obstacle configuration"""
        # Generate new configuration for this episode
        temp_config_path = self.static_wrapper.generate_episode_config()
        
        # Update environment kwargs with new world configuration
        updated_kwargs = self.env_kwargs.copy()
        updated_kwargs['world_name'] = temp_config_path
        
        # Create new environment instance
        import gym
        env = gym.make('mrnav-v1', **updated_kwargs)
        
        return env
    
    def reset(self, mode=0, **kwargs):
        """Reset environment with new random obstacles"""
        self._reset_count += 1
        
        # Create new environment with random obstacles
        if self.env is not None:
            del self.env  # Clean up previous environment
            
        self.env = self._create_env_with_obstacles()
        
        # Reset the environment
        obs_list = self.env.reset(mode=mode, **kwargs)
        
        return obs_list
    
    def step_ir(self, action, **kwargs):
        """Step the environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.step_ir(action, **kwargs)
    
    def render(self, **kwargs):
        """Render the environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.render(**kwargs)
    
    def reset_one(self, id):
        """Reset one robot"""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.reset_one(id)
    
    def show(self):
        """Show environment"""
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.show()
    
    @property
    def ir_gym(self):
        """Access to underlying ir_gym"""
        if self.env is None:
            # Create temporary environment to get ir_gym access
            self.env = self._create_env_with_obstacles()
        return self.env.ir_gym
    
    @property
    def observation_space(self):
        """Observation space"""
        if self.env is None:
            # Create temporary environment to get space info
            temp_env = self._create_env_with_obstacles()
            space = temp_env.observation_space
            del temp_env
            return space
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Action space"""
        if self.env is None:
            # Create temporary environment to get space info
            temp_env = self._create_env_with_obstacles()
            space = temp_env.action_space
            del temp_env
            return space
        return self.env.action_space
    
    def cleanup(self):
        """Cleanup resources"""
        if self.env is not None:
            del self.env
            self.env = None
        self.static_wrapper.cleanup_temp_files()
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()


def create_static_obstacle_env(base_world_config: str, **env_kwargs) -> StaticObstacleGymWrapper:
    """
    Create a gym environment with random static obstacles
    
    Args:
        base_world_config: Path to base world configuration
        **env_kwargs: Additional environment arguments
        
    Returns:
        Random static obstacle gym wrapper
    """
    # Create static environment wrapper
    static_wrapper = StaticObstacleEnvWrapper(
        base_world_config_path=base_world_config,
        obstacle_count=4,
        obstacle_size=1.0,
        min_obstacle_distance=2.0
    )
    
    # Create gym wrapper
    import gym
    env_wrapper = StaticObstacleGymWrapper(
        env_class=gym.make,
        static_env_wrapper=static_wrapper,
        **env_kwargs
    )
    
    return env_wrapper
