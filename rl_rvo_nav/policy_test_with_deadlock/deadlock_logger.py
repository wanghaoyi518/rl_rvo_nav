"""
Deadlock Debug Logger

This module provides comprehensive logging functionality for deadlock resolution debugging.
It supports different log levels, timestamp-based file naming, and structured logging.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np


class DeadlockLogger:
    """
    Comprehensive logger for deadlock resolution debugging.
    
    Features:
    - Timestamp-based log files
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR)
    - Structured logging for different components
    - Performance metrics tracking
    - Episode and step-level logging
    """
    
    def __init__(self, log_dir: str = "deadlock_logs", log_level: str = "DEBUG"):
        """
        Initialize the deadlock logger.
        
        Args:
            log_dir: Directory to store log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamp-based log file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"deadlock_debug_{timestamp}.log"
        
        # Setup logging
        self.logger = logging.getLogger(f"deadlock_debug_{timestamp}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S.%f'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Statistics tracking
        self.stats = {
            'episode': 0,
            'step': 0,
            'deadlock_detections': 0,
            'mode_switches': 0,
            'par_executions': 0,
            'par_successes': 0,
            'par_failures': 0,
            'total_agents': 0
        }
        
        # Episode-level data
        self.episode_data = {
            'deadlock_events': [],
            'mode_switches': [],
            'par_executions': [],
            'agent_states': {},
            'episode_deadlock_count': 0  # Add episode-level deadlock counter
        }
        
        self.logger.info(f"🔧 Deadlock Logger initialized. Log file: {self.log_file}")
    
    def log_episode_start(self, episode_num: int, num_agents: int, config: Dict = None):
        """Log episode start information."""
        self.stats['episode'] = episode_num
        self.stats['step'] = 0
        self.stats['total_agents'] = num_agents
        
        # Reset episode data
        self.episode_data = {
            'deadlock_events': [],
            'mode_switches': [],
            'par_executions': [],
            'agent_states': {},
            'episode_deadlock_count': 0  # Reset episode-level deadlock counter
        }
        
        self.logger.info(f"🎬 EPISODE START: Episode {episode_num}, Agents: {num_agents}")
        if config:
            self.logger.debug(f"📋 Episode config: {json.dumps(config, indent=2)}")
    
    def log_step_start(self, step_num: int, agent_states: Dict, neighbor_states: Dict):
        """Log step start information."""
        self.stats['step'] = step_num
        
        self.logger.debug(f"🔄 STEP START: Episode {self.stats['episode']}, Step {step_num}")
        
        # Log agent states for debugging
        for agent_id, state in agent_states.items():
            if agent_id not in self.episode_data['agent_states']:
                self.episode_data['agent_states'][agent_id] = []
            
            # Store essential state information
            position = state.get('position', [0, 0])
            velocity = state.get('velocity', [0, 0])
            goal = state.get('goal', [0, 0])
            
            # Calculate distance to goal safely
            try:
                if goal is not None and len(goal) == 2 and len(position) == 2:
                    distance_to_goal = np.linalg.norm(np.array(position) - np.array(goal))
                else:
                    distance_to_goal = 0
            except:
                distance_to_goal = 0
            
            state_summary = {
                'step': step_num,
                'position': position,
                'velocity': velocity,
                'velocity_magnitude': np.linalg.norm(velocity),
                'goal': goal,
                'distance_to_goal': distance_to_goal
            }
            self.episode_data['agent_states'][agent_id].append(state_summary)
    
    def log_deadlock_detection(self, agent_id: int, trigger_type: str, details: Dict):
        """Log deadlock detection event."""
        self.stats['deadlock_detections'] += 1
        self.episode_data['episode_deadlock_count'] += 1  # Increment episode-level counter
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_details = self._convert_to_serializable(details)
        
        event = {
            'step': self.stats['step'],
            'agent_id': agent_id,
            'trigger_type': trigger_type,
            'details': serializable_details,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data['deadlock_events'].append(event)
        
        self.logger.warning(f"🔴 DEADLOCK DETECTED: Agent {agent_id}, Trigger: {trigger_type}")
        self.logger.debug(f"   Details: {json.dumps(serializable_details, indent=2)}")
    
    def log_deadlock_check(self, agent_id: int, velocity: float, threshold: float, neighbor_count: int):
        """Log deadlock check details."""
        self.logger.debug(f"🔍 DEADLOCK CHECK: Agent {agent_id}, Velocity: {velocity:.3f}, Threshold: {threshold}, Neighbors: {neighbor_count}")
    
    def log_mode_switch(self, agent_id: int, old_mode: str, new_mode: str, reason: str = ""):
        """Log mode switching event."""
        self.stats['mode_switches'] += 1
        
        event = {
            'step': self.stats['step'],
            'agent_id': agent_id,
            'old_mode': old_mode,
            'new_mode': new_mode,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data['mode_switches'].append(event)
        
        self.logger.info(f"🔄 MODE SWITCH: Agent {agent_id}: {old_mode} → {new_mode}")
        if reason:
            self.logger.debug(f"   Reason: {reason}")
    
    def log_par_preparation(self, agent_id: int, participants: List[int], par_solution: Dict):
        """Log PAR preparation event."""
        self.logger.info(f"📋 PAR PREPARATION: Agent {agent_id}, Participants: {participants}")
        
        # Convert par_solution to serializable format
        try:
            serializable_solution = self._convert_to_serializable(par_solution)
            self.logger.debug(f"   PAR Solution: {json.dumps(serializable_solution, indent=2)}")
        except Exception as e:
            self.logger.debug(f"   PAR Solution: [Serialization failed: {str(e)}]")
    
    def log_par_execution(self, agent_id: int, action: np.ndarray, status: str):
        """Log PAR execution event."""
        self.stats['par_executions'] += 1
        
        event = {
            'step': self.stats['step'],
            'agent_id': agent_id,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data['par_executions'].append(event)
        
        if status == 'success':
            self.stats['par_successes'] += 1
            self.logger.info(f"✅ PAR EXECUTION: Agent {agent_id} executed successfully")
        elif status == 'failure':
            self.stats['par_failures'] += 1
            self.logger.error(f"❌ PAR EXECUTION: Agent {agent_id} failed")
        else:
            self.logger.debug(f"🔄 PAR EXECUTION: Agent {agent_id} - {status}")
    
    def log_par_completion(self, agent_id: int, total_steps: int):
        """Log PAR completion event."""
        self.logger.info(f"🎉 PAR COMPLETED: Agent {agent_id} finished in {total_steps} steps")
    
    def log_rl_to_mapf_positions(self, agent_positions: Dict[int, List[float]]):
        """Log agent positions when switching from RL to MAPF mode."""
        self.logger.info(f"📍 RL→MAPF POSITIONS: Agent positions when entering deadlock resolution")
        for agent_id, position in agent_positions.items():
            self.logger.info(f"   Agent {agent_id}: position={position}")
    
    def log_mapf_to_rl_positions(self, agent_positions: Dict[int, List[float]]):
        """Log agent positions when switching from MAPF back to RL mode."""
        self.logger.info(f"📍 MAPF→RL POSITIONS: Agent positions when exiting deadlock resolution")
        for agent_id, position in agent_positions.items():
            self.logger.info(f"   Agent {agent_id}: position={position}")
    
    def log_par_solution_paths(self, par_solution, participants: List[int]):
        """Log PAR solution paths for each participant."""
        self.logger.info(f"🗺️ PAR SOLUTION PATHS: Generated paths for {len(participants)} agents")
        
        if par_solution and hasattr(par_solution, 'agents_moves'):
            # Group moves by agent
            agent_moves = {}
            for move in par_solution.agents_moves:
                if move.id not in agent_moves:
                    agent_moves[move.id] = []
                agent_moves[move.id].append((move.di, move.dj))
            
            # Log paths for each participant
            for agent_id in participants:
                if agent_id in agent_moves:
                    moves = agent_moves[agent_id]
                    path_str = " → ".join([f"({di}, {dj})" for di, dj in moves])
                    self.logger.info(f"   Agent {agent_id}: {path_str}")
                else:
                    self.logger.info(f"   Agent {agent_id}: No path generated")
        else:
            self.logger.warning(f"   No valid PAR solution available")
    
    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log error events."""
        self.logger.error(f"❌ ERROR in {component}: {str(error)}")
        if context:
            self.logger.debug(f"   Context: {json.dumps(context, indent=2)}")
    
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics."""
        self.logger.info(f"📊 PERFORMANCE: {json.dumps(metrics, indent=2)}")
    
    def log_episode_summary(self):
        """Log episode summary."""
        summary = {
            'episode': self.stats['episode'],
            'total_steps': self.stats['step'],
            'deadlock_detections': self.stats['deadlock_detections'],
            'episode_deadlock_count': self.episode_data['episode_deadlock_count'],  # Add episode-level count
            'mode_switches': self.stats['mode_switches'],
            'par_executions': self.stats['par_executions'],
            'par_successes': self.stats['par_successes'],
            'par_failures': self.stats['par_failures'],
            'success_rate': self.stats['par_successes'] / max(self.stats['par_executions'], 1)
        }
        
        self.logger.info(f"📈 EPISODE SUMMARY: {json.dumps(summary, indent=2)}")
        
        # Log detailed episode data
        try:
            serializable_episode_data = self._convert_to_serializable(self.episode_data)
            self.logger.debug(f"📋 EPISODE DETAILS: {json.dumps(serializable_episode_data, indent=2)}")
        except Exception as e:
            self.logger.debug(f"📋 EPISODE DETAILS: [Serialization failed: {str(e)}]")
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.copy()
    
    def save_episode_data(self, filename: str = None):
        """Save episode data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_data_{self.stats['episode']}_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Convert data to serializable format
        serializable_data = {
            'stats': self.stats,
            'episode_data': self._convert_to_serializable(self.episode_data),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Episode data saved to: {filepath}")
        return filepath
    
    def log_par_solver_details(self, par_solver_input: Dict, par_solution, participants: List[int]):
        """Log detailed PAR solver information including initialization, trajectories, and final positions."""
        # Log PAR solver initialization details
        self.logger.info(f"🔧 PAR SOLVER DETAILS: Detailed execution information")
        
        # Log initialization positions
        if 'start_positions' in par_solver_input:
            self.logger.info(f"📍 PAR INITIALIZATION: Agent positions at start of PAR execution")
            for agent_id, pos in par_solver_input['start_positions'].items():
                self.logger.info(f"   Agent {agent_id}: start_position={pos}")
        
        # Log goal positions
        if 'goal_positions' in par_solver_input:
            self.logger.info(f"🎯 PAR GOALS: Agent goal positions for PAR execution")
            for agent_id, pos in par_solver_input['goal_positions'].items():
                self.logger.info(f"   Agent {agent_id}: goal_position={pos}")
        
        # Log computed trajectories
        if par_solution and hasattr(par_solution, 'agents_moves'):
            self.logger.info(f"🛤️ PAR TRAJECTORIES: Computed movement paths for each agent")
            
            # Group moves by agent
            agent_moves = {}
            for move in par_solution.agents_moves:
                if move.id not in agent_moves:
                    agent_moves[move.id] = []
                agent_moves[move.id].append((move.di, move.dj))
            
            # Log detailed trajectories
            for agent_id in participants:
                if agent_id in agent_moves:
                    moves = agent_moves[agent_id]
                    # Calculate cumulative path
                    path = []
                    current_pos = par_solver_input['start_positions'].get(agent_id, (0, 0))
                    path.append(current_pos)
                    
                    for di, dj in moves:
                        new_pos = (current_pos[0] + di, current_pos[1] + dj)
                        path.append(new_pos)
                        current_pos = new_pos
                    
                    path_str = " → ".join([f"({x}, {y})" for x, y in path])
                    self.logger.info(f"   Agent {agent_id}: {path_str}")
                else:
                    self.logger.info(f"   Agent {agent_id}: No trajectory computed")
        
        # Store detailed information in episode data for later analysis
        par_details = {
            'step': self.stats['step'],
            'initialization': {
                'start_positions': par_solver_input.get('start_positions', {}),
                'goal_positions': par_solver_input.get('goal_positions', {}),
                'participants': participants,
                'workspace_bounds': par_solver_input.get('workspace_bounds', {}),
                'grid_resolution': par_solver_input.get('grid_resolution', 0.2)
            },
            'solution': {
                'trajectories': {},
                'success': par_solution is not None,
                'solver_type': 'PNR' if par_solution and hasattr(par_solution, 'agents_moves') else 'Fallback'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract trajectories if available
        if par_solution and hasattr(par_solution, 'agents_moves'):
            agent_moves = {}
            for move in par_solution.agents_moves:
                if move.id not in agent_moves:
                    agent_moves[move.id] = []
                agent_moves[move.id].append((move.di, move.dj))
            
            for agent_id in participants:
                if agent_id in agent_moves:
                    moves = agent_moves[agent_id]
                    # Calculate cumulative path
                    path = []
                    current_pos = par_solver_input['start_positions'].get(agent_id, (0, 0))
                    path.append(current_pos)
                    
                    for di, dj in moves:
                        new_pos = (current_pos[0] + di, current_pos[1] + dj)
                        path.append(new_pos)
                        current_pos = new_pos
                    
                    par_details['solution']['trajectories'][agent_id] = {
                        'moves': moves,
                        'path': path,
                        'final_position': path[-1] if path else current_pos
                    }
                else:
                    par_details['solution']['trajectories'][agent_id] = {
                        'moves': [],
                        'path': [],
                        'final_position': par_solver_input['start_positions'].get(agent_id, (0, 0))
                    }
        
        # Store in episode data
        if 'par_solver_details' not in self.episode_data:
            self.episode_data['par_solver_details'] = []
        self.episode_data['par_solver_details'].append(par_details)
    
    def log_par_final_positions(self, agent_final_positions: Dict[int, List[float]]):
        """Log final agent positions after PAR execution completion."""
        self.logger.info(f"🏁 PAR FINAL POSITIONS: Agent positions after PAR execution completion")
        for agent_id, position in agent_final_positions.items():
            self.logger.info(f"   Agent {agent_id}: final_position={position}")
        
        # Store final positions in episode data
        if 'par_final_positions' not in self.episode_data:
            self.episode_data['par_final_positions'] = []
        
        final_positions_event = {
            'step': self.stats['step'],
            'agent_final_positions': agent_final_positions,
            'timestamp': datetime.now().isoformat()
        }
        self.episode_data['par_final_positions'].append(final_positions_event)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj


# Global logger instance
_deadlock_logger = None


def get_deadlock_logger(log_dir: str = "deadlock_logs", log_level: str = "DEBUG") -> DeadlockLogger:
    """Get or create global deadlock logger instance."""
    global _deadlock_logger
    if _deadlock_logger is None:
        _deadlock_logger = DeadlockLogger(log_dir, log_level)
    return _deadlock_logger


def reset_deadlock_logger():
    """Reset global deadlock logger instance."""
    global _deadlock_logger
    _deadlock_logger = None
