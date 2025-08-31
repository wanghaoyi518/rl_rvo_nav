"""
Configuration Usage Example

This file demonstrates how to use the DeadlockConfig class and its features.
"""

from .deadlock_config import DeadlockConfig, get_conservative_config, get_aggressive_config, get_balanced_config


def example_basic_usage():
    """Example of basic configuration usage."""
    print("=== Basic Configuration Usage ===")
    
    # Create default configuration
    config = DeadlockConfig()
    
    # Get specific configuration values
    small_speed = config.get('SMALL_SPEED')
    velocity_window = config.get('VELOCITY_WINDOW_SIZE')
    
    print(f"SMALL_SPEED: {small_speed}")
    print(f"VELOCITY_WINDOW_SIZE: {velocity_window}")
    
    # Set configuration values
    config.set('DEBUG_MODE', True)
    config.set('SMALL_SPEED', 0.15)
    
    print(f"Updated SMALL_SPEED: {config.get('SMALL_SPEED')}")
    print(f"DEBUG_MODE: {config.get('DEBUG_MODE')}")


def example_file_operations():
    """Example of configuration file operations."""
    print("\n=== Configuration File Operations ===")
    
    # Create configuration
    config = DeadlockConfig()
    
    # Save configuration to file
    config.save_to_file('my_config.json')
    
    # Load configuration from file
    new_config = DeadlockConfig('my_config.json')
    
    print("Configuration loaded from file successfully")
    print(f"SMALL_SPEED from file: {new_config.get('SMALL_SPEED')}")


def example_preset_configurations():
    """Example of using preset configurations."""
    print("\n=== Preset Configurations ===")
    
    # Get conservative configuration
    conservative_config = get_conservative_config()
    print("Conservative Configuration:")
    print(f"  SMALL_SPEED: {conservative_config.get('SMALL_SPEED')}")
    print(f"  VELOCITY_WINDOW_SIZE: {conservative_config.get('VELOCITY_WINDOW_SIZE')}")
    print(f"  MAPF_NUM: {conservative_config.get('MAPF_NUM')}")
    
    # Get aggressive configuration
    aggressive_config = get_aggressive_config()
    print("\nAggressive Configuration:")
    print(f"  SMALL_SPEED: {aggressive_config.get('SMALL_SPEED')}")
    print(f"  VELOCITY_WINDOW_SIZE: {aggressive_config.get('VELOCITY_WINDOW_SIZE')}")
    print(f"  MAPF_NUM: {aggressive_config.get('MAPF_NUM')}")
    
    # Get balanced configuration
    balanced_config = get_balanced_config()
    print("\nBalanced Configuration:")
    print(f"  SMALL_SPEED: {balanced_config.get('SMALL_SPEED')}")
    print(f"  VELOCITY_WINDOW_SIZE: {balanced_config.get('VELOCITY_WINDOW_SIZE')}")
    print(f"  MAPF_NUM: {balanced_config.get('MAPF_NUM')}")


def example_configuration_groups():
    """Example of getting configuration groups."""
    print("\n=== Configuration Groups ===")
    
    config = DeadlockConfig()
    
    # Get deadlock detection configuration
    deadlock_config = config.get_deadlock_detection_config()
    print("Deadlock Detection Configuration:")
    for key, value in deadlock_config.items():
        print(f"  {key}: {value}")
    
    # Get PAR algorithm configuration
    par_config = config.get_par_config()
    print("\nPAR Algorithm Configuration:")
    for key, value in par_config.items():
        print(f"  {key}: {value}")
    
    # Get mode switching configuration
    mode_config = config.get_mode_switching_config()
    print("\nMode Switching Configuration:")
    for key, value in mode_config.items():
        print(f"  {key}: {value}")


def example_configuration_validation():
    """Example of configuration validation."""
    print("\n=== Configuration Validation ===")
    
    # Create valid configuration
    config = DeadlockConfig()
    is_valid = config.validate_config()
    print(f"Default configuration is valid: {is_valid}")
    
    # Create invalid configuration
    invalid_config = DeadlockConfig()
    invalid_config.set('SMALL_SPEED', -1)  # Invalid value
    invalid_config.set('TRIGGER_TYPE', 'INVALID')  # Invalid trigger type
    
    is_valid = invalid_config.validate_config()
    print(f"Invalid configuration is valid: {is_valid}")


def example_configuration_summary():
    """Example of getting configuration summary."""
    print("\n=== Configuration Summary ===")
    
    config = DeadlockConfig()
    summary = config.get_config_summary()
    
    print("Configuration Summary:")
    for group_name, group_config in summary.items():
        print(f"\n{group_name.upper()}:")
        for key, value in group_config.items():
            print(f"  {key}: {value}")


def example_dynamic_configuration():
    """Example of dynamic configuration updates."""
    print("\n=== Dynamic Configuration Updates ===")
    
    # Create configuration
    config = DeadlockConfig()
    
    # Update multiple parameters at once
    updates = {
        'SMALL_SPEED': 0.08,
        'VELOCITY_WINDOW_SIZE': 75,
        'DEBUG_MODE': True,
        'LOG_LEVEL': 'DEBUG'
    }
    
    config.update_from_dict(updates)
    
    print("Updated Configuration:")
    print(f"  SMALL_SPEED: {config.get('SMALL_SPEED')}")
    print(f"  VELOCITY_WINDOW_SIZE: {config.get('VELOCITY_WINDOW_SIZE')}")
    print(f"  DEBUG_MODE: {config.get('DEBUG_MODE')}")
    print(f"  LOG_LEVEL: {config.get('LOG_LEVEL')}")
    
    # Reset to defaults
    config.reset_to_defaults()
    print(f"\nAfter reset - SMALL_SPEED: {config.get('SMALL_SPEED')}")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_file_operations()
    example_preset_configurations()
    example_configuration_groups()
    example_configuration_validation()
    example_configuration_summary()
    example_dynamic_configuration()
    
    print("\n=== All Examples Completed ===")
