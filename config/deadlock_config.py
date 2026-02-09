"""
Deadlock Configuration Module

This module provides configuration parameters for deadlock detection and PAR algorithm.
It contains all the configurable parameters used by the deadlock resolution system.
"""

from typing import Dict, Any
import os
import json


class DeadlockConfig:
    """
    Configuration class for deadlock detection and PAR algorithm.
    
    This class contains all configurable parameters for the deadlock resolution system,
    including detection thresholds, PAR parameters, and mode switching settings.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the deadlock configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        # Default configuration values
        self.default_config = {
            # Deadlock Detection Parameters
            'DEADLOCK_DETECTION_ENABLED': True,
            'TRIGGER_TYPE': 'SPEED_BUFFER',  # Use speed buffer trigger for deadlock detection
            'SMALL_SPEED': 0.3,  # Slightly higher to classify more agents as slow
            'VELOCITY_WINDOW_SIZE': 10,  # Number of time steps for velocity averaging (reduced from 50 to 20 for faster response)
            'MAPF_NUM': 4,  # Minimum number of agents to trigger PAR (reduced from 10 to 4 to match 8 robots)
            'SIGHT_RADIUS': 7.0,  # Radius for detecting nearby agents
            'EPISODE_START_DELAY': 0,  # Delay before starting deadlock detection (reduced from 50 to 5 for early detection)

            # Enhanced SPEED_BUFFER trigger thresholds
            'TTC_THRESHOLD': 3.0,              # Slightly relaxed TTC gate
            'DMIN_THRESHOLD': 0.6,             # Slightly relaxed Dmin gate
            'REQUIRED_NON_PROGRESS_NEIGHBORS': 1,  # neighbors count threshold

            # Waypoint-stuck independent trigger
            'ENABLE_WAYPOINT_STUCK_TRIGGER': True,
            'WAYPOINT_STUCK_STEPS': 100,       # steps before triggering (recommend >=100)

            # Waypoint force-switch (advance to next waypoint after N steps without reaching)
            'FORCE_WAYPOINT_SWITCH_ENABLED': True,
            'FORCE_WAYPOINT_SWITCH_STEPS': 20,
            
            
            # PAR Algorithm Parameters
            'PAR_OFFSET': 2,  # Offset for expanding PAR region
            'GRID_RESOLUTION': 0.5,  # Grid resolution for PAR environment (cell-centered visualization and planning)
            'USE_FULL_MAP': True,  # If True, discretize the full map instead of local sub-map
            'POSITION_TOLERANCE': 0.1,  # Tolerance for position matching
            'VELOCITY_SCALE': 1.0,  # Scale factor for velocity calculation
            'MAX_VELOCITY': 1.5,  # Maximum velocity limit
            
            # Mode Switching Parameters
            'MODE_SWITCH_DELAY': 20,  # Minimum time steps between mode switches (increased from 10 to reduce frequent switching)
            'PAR_COMPLETION_THRESHOLD': 15,  # Steps threshold for considering PAR path complete (increased from 10 for stability)
            'DEADLOCK_DETECTION_COOLDOWN': 0,  # Reduced from 100 to 20 for more frequent detection attempts  # Cooldown period between deadlock detections for the same agent (increased to reduce frequency)
            'GOAL_TOLERANCE': 0.5,  # Tolerance for reaching goal
            'NARROW_CORRIDOR_THRESHOLD': 4,  # Number of agents indicating narrow corridor
            'CONFINED_AREA_VELOCITY_THRESHOLD': 0.05,  # Velocity threshold for confined area
            'PARTICIPANT_LOCK_STEPS': 5,  # Lock participant set for N steps to avoid oscillation (Experiment B)
            
            # Communication and Coordination Parameters
            'COMMUNICATION_RANGE': 10.0,  # Communication range for agents (explicit per request)
            'COORDINATION_TIMEOUT': 100,  # Timeout for coordination operations
            
            # Performance and Debug Parameters
            'DEBUG_MODE': False,  # Enable debug output for PAR diagnostics
            'LOG_LEVEL': 'WARNING',  # Reduce logging level
            'SAVE_STATISTICS': False,  # Disable statistics saving to reduce I/O
            'STATISTICS_INTERVAL': 100,  # Interval for saving statistics
            
            # Advanced Parameters
            'ENABLE_DYNAMIC_PARTICIPANTS': True,  # Allow dynamic participant addition
            'MAX_PAR_PARTICIPANTS': 10,  # Maximum number of PAR participants
            'PAR_TIMEOUT': 1000,  # Timeout for PAR execution
            'ENABLE_POSITION_SYNC': True,  # Enable position synchronization after PAR

            # PAR execution interactions (RVO/neighbor/yielding)
            'EXCLUDE_PAR_NEIGHBORS_IN_RVO': False,      # Step 4 (set False to enable RVO mutual avoidance among PAR agents)
            'EXCLUDE_PAR_NEIGHBORS_IN_PIPELINE': True,  # Step 7
            'NON_PAR_YIELDING_ENABLED': True,           # Step 5
            'NON_PAR_YIELD_RADIUS': 2.0,
            'NON_PAR_YIELD_SPEED_SCALE': 0.5,
            'PAR_TRACK_SPEED_LIMIT': 0.3,
            'PAR_TRACK_SPEED_LIMIT_SINGLE': 0.6,        # Single-agent fallback: higher cap for faster tracking
            'PAR_AGENT_MAX_STEPS': 0,                  # 0 disables per-agent timeout

            # Multi-hop conflict graph parameters
            'MAX_GRAPH_HOPS': 2,               # BFS hops for local conflict graph (neighbor's neighbor)
            'MAX_CANDIDATES_PER_HOP': 12,      # Per-hop candidate cap
            'MAX_GRAPH_NODES': 20,             # Total graph node cap
            
            # MAPF solver parameters
            'PAR_MAX_STEPS': 2000,  # Maximum steps for MAPF solver
            'PAR_HEURISTIC_WEIGHT': 0.6,  # Heuristic weight for A* search
            
            # Single agent deadlock trigger parameters
            'SINGLE_AGENT_TRIGGER_ENABLED': True,  # Enable single-agent fallback trigger
            'SINGLE_AGENT_TIME_THRESHOLD': 50,  # Time threshold for single agent deadlock trigger
            'MAX_NEIGHBORS_FOR_SINGLE_TRIGGER': 1,  # Maximum number of active neighbors for single agent trigger

            # Single-agent environment and yielding
            'ENABLE_SINGLE_AGENT_CORRIDOR': True,              # Enable corridor crop for single-agent fallback
            'PAR_SINGLE_AGENT_CORRIDOR_RADIUS': 2.0,           # meters
            'ARRIVED_AGENTS_AS_OBSTACLES': True,               # Overlay arrived agents as static obstacles
            'ARRIVED_AGENT_INFLATION_CELLS': 1,                # cells
            'ARRIVED_GOAL_TOLERANCE': 0.3,                     # meters
            'SINGLE_AGENT_YIELD_RADIUS': 2.0,                  # meters
            'SINGLE_AGENT_YIELD_SPEED_SCALE': 0.4,             # (0,1]
        }
        
        # Load configuration from file if provided
        self.config = self.default_config.copy()
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
    
    def load_from_file(self, config_file: str):
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update configuration with file values
            self.config.update(file_config)
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Error loading configuration from {config_file}: {e}")
            print("Using default configuration")
    
    def save_to_file(self, config_file: str):
        """
        Save current configuration to a JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {config_file}")
            
        except Exception as e:
            print(f"Error saving configuration to {config_file}: {e}")
    
    def get_deadlock_detection_config(self) -> Dict:
        """
        Get configuration specific to deadlock detection.
        
        Returns:
            Dict: Deadlock detection configuration
        """
        return {
            'DEADLOCK_DETECTION_ENABLED': self.get('DEADLOCK_DETECTION_ENABLED'),
            'TRIGGER_TYPE': self.get('TRIGGER_TYPE'),
            'SMALL_SPEED': self.get('SMALL_SPEED'),
            'VELOCITY_WINDOW_SIZE': self.get('VELOCITY_WINDOW_SIZE'),
            'MAPF_NUM': self.get('MAPF_NUM'),
            'SIGHT_RADIUS': self.get('SIGHT_RADIUS'),
        }
    
    def get_par_config(self) -> Dict:
        """
        Get configuration specific to PAR algorithm.
        
        Returns:
            Dict: PAR algorithm configuration
        """
        return {
            'PAR_OFFSET': self.get('PAR_OFFSET'),
            'GRID_RESOLUTION': self.get('GRID_RESOLUTION'),
            'POSITION_TOLERANCE': self.get('POSITION_TOLERANCE'),
            'VELOCITY_SCALE': self.get('VELOCITY_SCALE'),
            'MAX_VELOCITY': self.get('MAX_VELOCITY'),
            'MAX_PAR_PARTICIPANTS': self.get('MAX_PAR_PARTICIPANTS'),
            'PAR_TIMEOUT': self.get('PAR_TIMEOUT'),
            'ENABLE_POSITION_SYNC': self.get('ENABLE_POSITION_SYNC'),
        }
    
    def get_mode_switching_config(self) -> Dict:
        """
        Get configuration specific to mode switching.
        
        Returns:
            Dict: Mode switching configuration
        """
        return {
            'MODE_SWITCH_DELAY': self.get('MODE_SWITCH_DELAY'),
            'PAR_COMPLETION_THRESHOLD': self.get('PAR_COMPLETION_THRESHOLD'),
            'GOAL_TOLERANCE': self.get('GOAL_TOLERANCE'),
            'NARROW_CORRIDOR_THRESHOLD': self.get('NARROW_CORRIDOR_THRESHOLD'),
            'CONFINED_AREA_VELOCITY_THRESHOLD': self.get('CONFINED_AREA_VELOCITY_THRESHOLD'),
        }
    
    def get_communication_config(self) -> Dict:
        """
        Get configuration specific to communication and coordination.
        
        Returns:
            Dict: Communication configuration
        """
        return {
            'COMMUNICATION_RANGE': self.get('COMMUNICATION_RANGE'),
            'COORDINATION_TIMEOUT': self.get('COORDINATION_TIMEOUT'),
            'ENABLE_DYNAMIC_PARTICIPANTS': self.get('ENABLE_DYNAMIC_PARTICIPANTS'),
        }
    
    def get_debug_config(self) -> Dict:
        """
        Get configuration specific to debugging and performance monitoring.
        
        Returns:
            Dict: Debug configuration
        """
        return {
            'DEBUG_MODE': self.get('DEBUG_MODE'),
            'LOG_LEVEL': self.get('LOG_LEVEL'),
            'SAVE_STATISTICS': self.get('SAVE_STATISTICS'),
            'STATISTICS_INTERVAL': self.get('STATISTICS_INTERVAL'),
        }
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            bool: True if configuration is valid
        """
        errors = []
        
        # Check required parameters
        required_params = [
            'SMALL_SPEED', 'VELOCITY_WINDOW_SIZE', 'MAPF_NUM', 'SIGHT_RADIUS',
            'PAR_OFFSET', 'GRID_RESOLUTION', 'POSITION_TOLERANCE',
            'MODE_SWITCH_DELAY', 'GOAL_TOLERANCE'
        ]
        
        for param in required_params:
            if param not in self.config:
                errors.append(f"Missing required parameter: {param}")
            elif self.config[param] is None:
                errors.append(f"Parameter {param} cannot be None")
        
        # Check parameter ranges
        if self.get('SMALL_SPEED', 0) <= 0:
            errors.append("SMALL_SPEED must be positive")
        
        if self.get('VELOCITY_WINDOW_SIZE', 0) <= 0:
            errors.append("VELOCITY_WINDOW_SIZE must be positive")
        
        if self.get('MAPF_NUM', 0) < 1:
            errors.append("MAPF_NUM must be at least 1")
        
        if self.get('SIGHT_RADIUS', 0) <= 0:
            errors.append("SIGHT_RADIUS must be positive")
        
        if self.get('GRID_RESOLUTION', 0) <= 0:
            errors.append("GRID_RESOLUTION must be positive")
        
        if self.get('POSITION_TOLERANCE', 0) <= 0:
            errors.append("POSITION_TOLERANCE must be positive")
        
        if self.get('MODE_SWITCH_DELAY', 0) < 0:
            errors.append("MODE_SWITCH_DELAY must be non-negative")
        
        if self.get('GOAL_TOLERANCE', 0) <= 0:
            errors.append("GOAL_TOLERANCE must be positive")

        # New: Enhanced SPEED_BUFFER gates
        if self.get('TTC_THRESHOLD', 0) <= 0:
            errors.append("TTC_THRESHOLD must be positive")
        if self.get('DMIN_THRESHOLD', 0) <= 0:
            errors.append("DMIN_THRESHOLD must be positive")
        if self.get('REQUIRED_NON_PROGRESS_NEIGHBORS', -1) < 0:
            errors.append("REQUIRED_NON_PROGRESS_NEIGHBORS must be non-negative")

        # New: PAR interaction controls
        if self.get('NON_PAR_YIELD_RADIUS', -1) < 0:
            errors.append("NON_PAR_YIELD_RADIUS must be non-negative")
        s = self.get('NON_PAR_YIELD_SPEED_SCALE', 0)
        if s <= 0 or s > 1:
            errors.append("NON_PAR_YIELD_SPEED_SCALE must be in (0, 1]")
        if self.get('PAR_TRACK_SPEED_LIMIT', 0) <= 0:
            errors.append("PAR_TRACK_SPEED_LIMIT must be positive")
        if self.get('PAR_TRACK_SPEED_LIMIT_SINGLE', 0) <= 0:
            errors.append("PAR_TRACK_SPEED_LIMIT_SINGLE must be positive")
        if self.get('PAR_AGENT_MAX_STEPS', -1) < 0:
            errors.append("PAR_AGENT_MAX_STEPS must be >= 0 (0 disables timeout)")
        
        # New: multi-hop graph parameters
        if int(self.get('MAX_GRAPH_HOPS', 0)) <= 0:
            errors.append("MAX_GRAPH_HOPS must be positive")
        if int(self.get('MAX_CANDIDATES_PER_HOP', 0)) <= 0:
            errors.append("MAX_CANDIDATES_PER_HOP must be positive")
        if int(self.get('MAX_GRAPH_NODES', 0)) <= 0:
            errors.append("MAX_GRAPH_NODES must be positive")

        # Waypoint force-switch parameters
        if not isinstance(self.get('FORCE_WAYPOINT_SWITCH_ENABLED', True), bool):
            errors.append("FORCE_WAYPOINT_SWITCH_ENABLED must be boolean")
        if int(self.get('FORCE_WAYPOINT_SWITCH_STEPS', 0)) <= 0:
            errors.append("FORCE_WAYPOINT_SWITCH_STEPS must be positive")

        # Single-agent fallback and environment parameters
        if not isinstance(self.get('SINGLE_AGENT_TRIGGER_ENABLED', True), bool):
            errors.append("SINGLE_AGENT_TRIGGER_ENABLED must be boolean")
        if float(self.get('PAR_SINGLE_AGENT_CORRIDOR_RADIUS', 0)) <= 0:
            errors.append("PAR_SINGLE_AGENT_CORRIDOR_RADIUS must be positive")
        if int(self.get('ARRIVED_AGENT_INFLATION_CELLS', -1)) < 0:
            errors.append("ARRIVED_AGENT_INFLATION_CELLS must be >= 0")
        yscale = float(self.get('SINGLE_AGENT_YIELD_SPEED_SCALE', 0))
        if yscale <= 0 or yscale > 1:
            errors.append("SINGLE_AGENT_YIELD_SPEED_SCALE must be in (0, 1]")
        
        # Check hybrid detection parameters
        if self.get('TRIGGER_TYPE') == 'HYBRID':
            hybrid_mode = self.get('HYBRID_MODE')
            if hybrid_mode not in ['AND', 'OR']:
                errors.append("HYBRID_MODE must be 'AND' or 'OR'")
            
            speed_buffer_weight = self.get('SPEED_BUFFER_WEIGHT', 0)
            common_point_weight = self.get('COMMON_POINT_WEIGHT', 0)
            if speed_buffer_weight < 0 or speed_buffer_weight > 1:
                errors.append("SPEED_BUFFER_WEIGHT must be between 0 and 1")
            if common_point_weight < 0 or common_point_weight > 1:
                errors.append("COMMON_POINT_WEIGHT must be between 0 and 1")
            if abs(speed_buffer_weight + common_point_weight - 1.0) > 0.01:
                errors.append("SPEED_BUFFER_WEIGHT + COMMON_POINT_WEIGHT must sum to 1.0")
        
        # Check enhanced speed buffer parameters
        speed_buffer_avg_threshold = self.get('SPEED_BUFFER_AVG_THRESHOLD', 0)
        speed_buffer_max_threshold = self.get('SPEED_BUFFER_MAX_THRESHOLD', 0)
        if speed_buffer_avg_threshold <= 0:
            errors.append("SPEED_BUFFER_AVG_THRESHOLD must be positive")
        if speed_buffer_max_threshold <= 0:
            errors.append("SPEED_BUFFER_MAX_THRESHOLD must be positive")
        if speed_buffer_avg_threshold >= speed_buffer_max_threshold:
            errors.append("SPEED_BUFFER_AVG_THRESHOLD must be less than SPEED_BUFFER_MAX_THRESHOLD")
        
        speed_buffer_min_history_ratio = self.get('SPEED_BUFFER_MIN_HISTORY_RATIO', 0)
        if speed_buffer_min_history_ratio <= 0 or speed_buffer_min_history_ratio > 1:
            errors.append("SPEED_BUFFER_MIN_HISTORY_RATIO must be between 0 and 1")
        
        # Check trigger type: only SPEED_BUFFER is supported
        trigger_type = self.get('TRIGGER_TYPE')
        if trigger_type not in ['SPEED_BUFFER']:
            errors.append("TRIGGER_TYPE must be 'SPEED_BUFFER'")
        
        # Report errors
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def get_config_summary(self) -> Dict:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dict: Configuration summary
        """
        return {
            'deadlock_detection': self.get_deadlock_detection_config(),
            'par_algorithm': self.get_par_config(),
            'mode_switching': self.get_mode_switching_config(),
            'communication': self.get_communication_config(),
            'debug': self.get_debug_config(),
        }
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self.default_config.copy()
    
    def update_from_dict(self, config_dict: Dict):
        """
        Update configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration updates
        """
        self.config.update(config_dict)
    
    def create_preset_config(self, preset_name: str) -> Dict:
        """
        Create a preset configuration for different scenarios.
        
        Args:
            preset_name: Name of the preset ('conservative', 'aggressive', 'balanced')
            
        Returns:
            Dict: Preset configuration
        """
        presets = {
            'conservative': {
                'SMALL_SPEED': 0.05,  # Lower threshold for early detection
                'VELOCITY_WINDOW_SIZE': 100,  # Longer window for stability
                'MAPF_NUM': 2,  # Trigger PAR with fewer agents
                'MODE_SWITCH_DELAY': 20,  # Longer delay between switches
                'PAR_OFFSET': 3,  # Larger PAR region
                'DEBUG_MODE': True,
            },
            'aggressive': {
                'SMALL_SPEED': 0.2,  # Higher threshold for late detection
                'VELOCITY_WINDOW_SIZE': 25,  # Shorter window for responsiveness
                'MAPF_NUM': 4,  # Trigger PAR with more agents
                'MODE_SWITCH_DELAY': 5,  # Shorter delay between switches
                'PAR_OFFSET': 1,  # Smaller PAR region
                'DEBUG_MODE': False,
            },
            'balanced': {
                'SMALL_SPEED': 0.1,  # Balanced threshold
                'VELOCITY_WINDOW_SIZE': 50,  # Balanced window
                'MAPF_NUM': 3,  # Balanced trigger condition
                'MODE_SWITCH_DELAY': 10,  # Balanced delay
                'PAR_OFFSET': 2,  # Balanced region size
                'DEBUG_MODE': False,
            }
        }
        
        if preset_name in presets:
            return presets[preset_name]
        else:
            print(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
            return {}
    
    def apply_preset(self, preset_name: str):
        """
        Apply a preset configuration.
        
        Args:
            preset_name: Name of the preset to apply
        """
        preset_config = self.create_preset_config(preset_name)
        if preset_config:
            self.update_from_dict(preset_config)
            print(f"Applied preset configuration: {preset_name}")


# Default configuration instance
default_config = DeadlockConfig()

# Preset configurations
def get_conservative_config() -> DeadlockConfig:
    """Get conservative configuration preset."""
    config = DeadlockConfig()
    config.apply_preset('conservative')
    return config

def get_aggressive_config() -> DeadlockConfig:
    """Get aggressive configuration preset."""
    config = DeadlockConfig()
    config.apply_preset('aggressive')
    return config

def get_balanced_config() -> DeadlockConfig:
    """Get balanced configuration preset."""
    config = DeadlockConfig()
    config.apply_preset('balanced')
    return config
