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
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    
    def __init__(self, log_dir: str = None, log_level: str = "DEBUG"):
        """
        Initialize the deadlock logger.
        
        Args:
            log_dir: Directory to store log files. If None, uses fixed path.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        if log_dir is None:
            # Use fixed absolute path to ensure consistent log location
            log_dir = "/home/haoyiwang/Desktop/RL_RVO/rl_rvo_nav/rl_rvo_nav/policy_test_with_deadlock/deadlock_logs"
        
        # Create a per-run timestamped subdirectory under the base log directory
        base_log_dir = Path(log_dir)
        base_log_dir.mkdir(exist_ok=True)
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_dir = base_log_dir / run_timestamp
        run_log_dir.mkdir(exist_ok=True)
        
        # Point logger directory to the per-run subdirectory so all artifacts go inside it
        self.log_dir = run_log_dir
        
        # Create timestamp-based log file name inside the run directory
        self.log_file = self.log_dir / f"deadlock_debug_{run_timestamp}.log"
        
        # Setup logging
        self.logger = logging.getLogger(f"deadlock_debug_{run_timestamp}")
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
        
        # Statistics tracking (PAR and CBS are distinguished)
        self.stats = {
            'episode': 0,
            'step': 0,
            'deadlock_detections': 0,
            'mode_switches': 0,
            'par_executions': 0,
            'par_successes': 0,
            'par_failures': 0,
            'cbs_executions': 0,
            'cbs_successes': 0,
            'cbs_failures': 0,
            'total_agents': 0
        }
        
        # Episode-level data
        self.episode_data = {
            'deadlock_events': [],
            'mode_switches': [],
            'par_executions': [],
            'cbs_executions': [],
            'agent_states': {},
            'episode_deadlock_count': 0
        }
        self.session_summaries = []
        
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
            'cbs_executions': [],
            'agent_states': {},
            'episode_deadlock_count': 0,
            'episode_config': config or {}
        }
        
        # self.logger.info(f"🎬 EPISODE START: Episode {episode_num}, Agents: {num_agents}")
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

    def log_deadlock_participants(self, agent_id: int, participants: List[int]):
        """Log deadlock participant list."""
        event = {
            'step': self.stats['step'],
            'agent_id': agent_id,
            'participants': participants,
            'count': len(participants),
            'timestamp': datetime.now().isoformat()
        }
        if 'deadlock_participants' not in self.episode_data:
            self.episode_data['deadlock_participants'] = []
        self.episode_data['deadlock_participants'].append(event)
    
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
        
        # self.logger.info(f"🔄 MODE SWITCH: Agent {agent_id}: {old_mode} → {new_mode}")
        pass
        if reason:
            self.logger.debug(f"   Reason: {reason}")
    
    def log_par_preparation(self, agent_id: int, participants: List[int], par_solution: Dict):
        """Log PAR preparation event."""
        # self.logger.info(f"📋 PAR PREPARATION: Agent {agent_id}, Participants: {participants}")
        pass
        
        # Convert par_solution to serializable format
        try:
            serializable_solution = self._convert_to_serializable(par_solution)
            self.logger.debug(f"   PAR Solution: {json.dumps(serializable_solution, indent=2)}")
        except Exception as e:
            self.logger.debug(f"   PAR Solution: [Serialization failed: {str(e)}]")
    
    def log_par_execution(self, agent_id: int, action: np.ndarray, status: str):
        """Log PAR execution event (per-agent step during PAR tracking)."""
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
        elif status == 'failure':
            self.stats['par_failures'] += 1
        else:
            self.logger.debug(f"PAR EXECUTION: Agent {agent_id} - {status}")

    def log_mapf_execution(self, solver_type: str, participants: List[int], status: str = 'success'):
        """Log one MAPF resolution event (PAR or CBS). Called when waypoints are injected after deadlock.
        solver_type: 'par' or 'cbs'
        """
        solver_type = (solver_type or 'par').lower()
        if solver_type == 'cbs':
            self.stats['cbs_executions'] += 1
            if status == 'success':
                self.stats['cbs_successes'] += 1
            elif status == 'failure':
                self.stats['cbs_failures'] += 1
            event = {
                'step': self.stats['step'],
                'solver': 'cbs',
                'participants': list(participants),
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            self.episode_data.setdefault('cbs_executions', []).append(event)
        else:
            self.stats['par_executions'] += 1
            if status == 'success':
                self.stats['par_successes'] += 1
            elif status == 'failure':
                self.stats['par_failures'] += 1
            event = {
                'step': self.stats['step'],
                'solver': 'par',
                'participants': list(participants),
                'status': status,
                'timestamp': datetime.now().isoformat()
            }
            self.episode_data['par_executions'].append(event)
    
    def log_par_completion(self, agent_id: int, total_steps: int):
        """Log PAR completion event."""
        # self.logger.info(f"🎉 PAR COMPLETED: Agent {agent_id} finished in {total_steps} steps")
        pass
    
    def log_rl_to_mapf_positions(self, agent_positions: Dict[int, List[float]]):
        """Log agent positions when switching from RL to MAPF mode."""
        # self.logger.info(f"📍 RL→MAPF POSITIONS: Agent positions when entering deadlock resolution")
        pass
        for agent_id, position in agent_positions.items():
            # self.logger.info(f"   Agent {agent_id}: position={position}")
            pass
    
    def log_mapf_to_rl_positions(self, agent_positions: Dict[int, List[float]]):
        """Log agent positions when switching from MAPF back to RL mode."""
        # self.logger.info(f"📍 MAPF→RL POSITIONS: Agent positions when exiting deadlock resolution")
        pass
        for agent_id, position in agent_positions.items():
            # self.logger.info(f"   Agent {agent_id}: position={position}")
            pass
    
    def log_par_solution_paths(self, par_solution, participants: List[int]):
        """Log PAR solution paths for each participant."""
        # self.logger.info(f"🗺️ PAR SOLUTION PATHS: Generated paths for {len(participants)} agents")
        pass
        
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
                    # self.logger.info(f"   Agent {agent_id}: {path_str}")
                    pass
                else:
                    # self.logger.info(f"   Agent {agent_id}: No path generated")
                    pass
        else:
            # self.logger.warning(f"   No valid PAR solution available")
            pass
    
    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log error events."""
        self.logger.error(f"❌ ERROR in {component}: {str(error)}")
        if context:
            self.logger.debug(f"   Context: {json.dumps(context, indent=2)}")
    
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics."""
        # self.logger.info(f"📊 PERFORMANCE: {json.dumps(metrics, indent=2)}")
        pass
    
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

        deadlock_events = self.episode_data.get('deadlock_events', []) or []
        mode_switches = self.episode_data.get('mode_switches', []) or []
        par_execs = self.episode_data.get('par_executions', []) or []
        participants_events = self.episode_data.get('deadlock_participants', []) or []
        episode_cfg = self.episode_data.get('episode_config', {}) or {}
        try:
            step_time = float(episode_cfg.get('step_time', 1.0))
        except Exception:
            step_time = 1.0

        detections_by_agent = {}
        for e in deadlock_events:
            aid = e.get('agent_id', None)
            if aid is None:
                continue
            detections_by_agent[aid] = detections_by_agent.get(aid, 0) + 1

        participants_hist = {}
        for e in participants_events:
            cnt = int(e.get('count', 0))
            participants_hist[cnt] = participants_hist.get(cnt, 0) + 1

        resolution_steps = []
        for e in deadlock_events:
            aid = e.get('agent_id', None)
            det_step = e.get('step', None)
            if aid is None or det_step is None:
                continue
            for sw in mode_switches:
                if sw.get('agent_id', None) != aid:
                    continue
                if sw.get('new_mode', '') != 'rl_rvo':
                    continue
                sw_step = sw.get('step', None)
                if sw_step is None:
                    continue
                if int(sw_step) >= int(det_step):
                    resolution_steps.append(int(sw_step) - int(det_step))
                    break
        resolution_seconds = [float(s) * step_time for s in resolution_steps]

        par_successes = 0
        par_failures = 0
        for e in par_execs:
            status = e.get('status', '')
            if status == 'success':
                par_successes += 1
            elif status == 'failure':
                par_failures += 1
        par_executions = len(par_execs)
        par_success_rate = float(par_successes) / float(par_executions) if par_executions > 0 else 0.0

        cbs_execs = self.episode_data.get('cbs_executions', []) or []
        cbs_successes = sum(1 for e in cbs_execs if e.get('status') == 'success')
        cbs_failures = sum(1 for e in cbs_execs if e.get('status') == 'failure')
        cbs_executions = len(cbs_execs)
        cbs_success_rate = float(cbs_successes) / float(cbs_executions) if cbs_executions > 0 else 0.0

        mapf_runtime_samples = []
        for detail in self.episode_data.get('par_solver_details', []) or []:
            meta = detail.get('solution', {}).get('meta', {}) if isinstance(detail, dict) else {}
            if isinstance(meta, dict):
                if isinstance(meta.get('runtime_wall'), (int, float)):
                    mapf_runtime_samples.append(float(meta.get('runtime_wall')))
                elif isinstance(meta.get('runtime'), (int, float)):
                    mapf_runtime_samples.append(float(meta.get('runtime')))

        def _stats(values):
            if not values:
                return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
            arr = np.array(values, dtype=float)
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95))
            }

        summary.update({
            "deadlock_events_total": len(deadlock_events),
            "deadlock_events_per_agent": detections_by_agent,
            "deadlock_participants_hist": participants_hist,
            "resolution_time_sec_samples": resolution_seconds,
            "resolution_time_sec_stats": _stats(resolution_seconds),
            "par_executions": par_executions,
            "par_successes": par_successes,
            "par_failures": par_failures,
            "par_success_rate": par_success_rate,
            "cbs_executions": cbs_executions,
            "cbs_successes": cbs_successes,
            "cbs_failures": cbs_failures,
            "cbs_success_rate": cbs_success_rate,
            "mapf_runtime_sec_samples": mapf_runtime_samples,
            "mapf_runtime_sec_stats": _stats(mapf_runtime_samples)
        })
        self.session_summaries.append(summary)
        
        # self.logger.info(f"📈 EPISODE SUMMARY: {json.dumps(summary, indent=2)}")
        
        # Log detailed episode data
        try:
            serializable_episode_data = self._convert_to_serializable(self.episode_data)
            self.logger.debug(f"📋 EPISODE DETAILS: {json.dumps(serializable_episode_data, indent=2)}")
        except Exception as e:
            self.logger.debug(f"📋 EPISODE DETAILS: [Serialization failed: {str(e)}]")

    def get_session_metrics(self) -> Dict:
        """Aggregate session-level deadlock and MAPF metrics."""
        summaries = self.session_summaries or []
        if not summaries:
            return {}

        total_episodes = len(summaries)
        episodes_with_deadlock = sum(1 for s in summaries if s.get("deadlock_events_total", 0) > 0)
        total_deadlock_events = sum(int(s.get("deadlock_events_total", 0) or 0) for s in summaries)

        all_resolution_samples = []
        all_mapf_runtime_samples = []
        participants_hist = {}

        total_par_exec = 0
        total_par_success = 0
        total_par_fail = 0
        total_cbs_exec = 0
        total_cbs_success = 0
        total_cbs_fail = 0

        for s in summaries:
            all_resolution_samples.extend(s.get("resolution_time_sec_samples", []) or [])
            all_mapf_runtime_samples.extend(s.get("mapf_runtime_sec_samples", []) or [])
            hist = s.get("deadlock_participants_hist", {}) or {}
            for k, v in hist.items():
                participants_hist[int(k)] = participants_hist.get(int(k), 0) + int(v)
            total_par_exec += int(s.get("par_executions", 0) or 0)
            total_par_success += int(s.get("par_successes", 0) or 0)
            total_par_fail += int(s.get("par_failures", 0) or 0)
            total_cbs_exec += int(s.get("cbs_executions", 0) or 0)
            total_cbs_success += int(s.get("cbs_successes", 0) or 0)
            total_cbs_fail += int(s.get("cbs_failures", 0) or 0)

        def _stats(values):
            if not values:
                return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
            arr = np.array(values, dtype=float)
            return {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95))
            }

        total_mode_switches = sum(int(s.get("mode_switches", 0) or 0) for s in summaries)
        return {
            "deadlock_episode_rate": float(episodes_with_deadlock) / float(total_episodes),
            "deadlock_events_per_episode": float(total_deadlock_events) / float(total_episodes),
            "total_deadlock_events": total_deadlock_events,
            "total_mode_switches": total_mode_switches,
            "deadlock_participants_hist": participants_hist,
            "resolution_time_sec": _stats(all_resolution_samples),
            "mapf_runtime_sec": _stats(all_mapf_runtime_samples),
            "par_success_rate": float(total_par_success) / float(total_par_exec) if total_par_exec > 0 else 0.0,
            "par_executions": total_par_exec,
            "par_successes": total_par_success,
            "par_failures": total_par_fail,
            "cbs_success_rate": float(total_cbs_success) / float(total_cbs_exec) if total_cbs_exec > 0 else 0.0,
            "cbs_executions": total_cbs_exec,
            "cbs_successes": total_cbs_success,
            "cbs_failures": total_cbs_fail
        }
    
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
        
        # self.logger.info(f"💾 Episode data saved to: {filepath}")
        return filepath
    
    def log_par_solver_details(self, par_solver_input: Dict, par_solution, participants: List[int]):
        """Log detailed PAR solver information including initialization, trajectories, and final positions."""
        # Log PAR solver initialization details
        # self.logger.info(f"🔧 PAR SOLVER DETAILS: Detailed execution information")
        pass

        # Ensure par_details is defined before any later references
        par_details = {
            'step': self.stats['step'],
            'initialization': {
                'start_positions': par_solver_input.get('start_positions', {}),
                'goal_positions': par_solver_input.get('goal_positions', {}),
                'participants': participants,
                'workspace_bounds': par_solver_input.get('workspace_bounds', {}),
                'grid_resolution': par_solver_input.get('grid_resolution', 0.2),
                'sub_map_dims': par_solver_input.get('sub_map_dims'),
                'obstacle_occupancy': par_solver_input.get('obstacle_occupancy'),
                'diagnostics': par_solver_input.get('diagnostics'),
                'participant_count': par_solver_input.get('participant_count'),
                'bfs_connectivity': par_solver_input.get('bfs_connectivity'),
                'connectivity_extras': par_solver_input.get('connectivity_extras'),
                'grid_offset': par_solver_input.get('grid_offset', (0, 0)),  # Add grid_offset for debugging
                'mapf_config': par_solver_input.get('mapf_config', {}),  # Add MAPF configuration
                'solver_actor_set': par_solver_input.get('solver_actor_set', []),  # Add solver actor set
                'id_mapping': par_solver_input.get('id_mapping', {}),  # Add ID mapping
                'sub_map_info': par_solver_input.get('sub_map_info', {}),  # Add complete SubMap information
                # map/agents summary (map_config_example-like)
                # 'grid_numeric': None,  # Removed to reduce log size
                'grid_pretty': None,
                # 'agents_summary': None,  # Removed to reduce log size
            },
            'solution': {
                'trajectories': {},
                'success': bool(getattr(par_solution, 'success', False)),
                'solver_type': 'PNR',
                'meta': {}
            },
            'timestamp': datetime.now().isoformat()
        }

        # Fill map/agents dump if sub_map available
        try:
            sub_map = None
            if isinstance(par_solver_input, dict):
                sub_map = par_solver_input.get('sub_map')
            if sub_map is not None and hasattr(sub_map, 'grid'):
                grid = sub_map.grid
                # Keep original grid format: grid[i][j] where i=row, j=col
                # This matches PNR algorithm's expectation: sub_map.grid[i][j]
                # grid_numeric = [[int(cell) for cell in row] for row in grid]  # Removed to reduce log size
                # par_details['initialization']['grid_numeric'] = grid_numeric  # Removed to reduce log size
                # pretty rows with indices (limited formatting) - reverse order for better readability
                rows = []
                for i, row in enumerate(grid):
                    rows.append(f"row{i}: {[int(cell) for cell in row]}")
                # Reverse the order so row0 appears at the bottom (like a normal coordinate system)
                par_details['initialization']['grid_pretty'] = list(reversed(rows))
                # Build MAP_CONFIG structure
                # Agents use (x, y) = (col, row) coordinate order to match RL coordinate system
                starts = par_solver_input.get('start_positions', {}) if isinstance(par_solver_input, dict) else {}
                goals = par_solver_input.get('goal_positions', {}) if isinstance(par_solver_input, dict) else {}
                agents_cfg = []
                for aid in participants:
                    sp = starts.get(aid)
                    gp = goals.get(aid)
                    if isinstance(sp, (list, tuple)) and len(sp) == 2 and isinstance(gp, (list, tuple)) and len(gp) == 2:
                        # sp = (x, y) = (col, row) - keep original order
                        agents_cfg.append({
                            'id': int(aid),
                            'start': [int(sp[0]), int(sp[1])],  # [x, y] = [col, row]
                            'goal': [int(gp[0]), int(gp[1])]    # [x, y] = [col, row]
                        })
                par_details['initialization']['MAP_CONFIG'] = {
                    'grid': [[int(cell) for cell in row] for row in grid],  # Generate grid on demand
                    'agents': agents_cfg
                }
        except Exception:
            pass

        # Build agents summary similar to test files - REMOVED to reduce log size
        # try:
        #     starts = par_solver_input.get('start_positions', {}) if isinstance(par_solver_input, dict) else {}
        #     goals = par_solver_input.get('goal_positions', {}) if isinstance(par_solver_input, dict) else {}
        #     meta = par_solver_input.get('solution_meta', {}) if isinstance(par_solver_input, dict) else {}
        #     st_ok = meta.get('starts_traversable', {}) if isinstance(meta, dict) else {}
        #     gt_ok = meta.get('goals_traversable', {}) if isinstance(meta, dict) else {}
        #     agents_summary = []
        #     for aid in participants:
        #         s = starts.get(aid)
        #         g = goals.get(aid)
        #         agents_summary.append({
        #             'agent_id': aid,
        #             'start_grid': list(s) if isinstance(s, (list, tuple)) else s,
        #             'goal_grid': list(g) if isinstance(g, (list, tuple)) else g,
        #             'start_traversable': st_ok.get(aid) if isinstance(st_ok, dict) else None,
        #             'goal_traversable': gt_ok.get(aid) if isinstance(gt_ok, dict) else None,
        #         })
        #     par_details['initialization']['agents_summary'] = agents_summary
        # except Exception:
        #     pass
        
        # Log initialization positions
        if 'start_positions' in par_solver_input:
            # self.logger.info(f"📍 PAR INITIALIZATION: Agent positions at start of PAR execution")
            pass
            for agent_id, pos in par_solver_input['start_positions'].items():
                # self.logger.info(f"   Agent {agent_id}: start_position={pos}")
                pass
        
        # Log goal positions
        if 'goal_positions' in par_solver_input:
            # self.logger.info(f"🎯 PAR GOALS: Agent goal positions for PAR execution")
            pass
            for agent_id, pos in par_solver_input['goal_positions'].items():
                # self.logger.info(f"   Agent {agent_id}: goal_position={pos}")
                pass
        
        # Log computed trajectories
        if par_solution and hasattr(par_solution, 'agents_moves'):
            # self.logger.info(f"🛤️ PAR TRAJECTORIES: Computed movement paths for each agent")
            pass
            
            # Group moves by agent
            agent_moves = {}
            for move in par_solution.agents_moves:
                if move.id not in agent_moves:
                    agent_moves[move.id] = []
                agent_moves[move.id].append((move.di, move.dj))
            
            # Log detailed trajectories and store in par_details
            for agent_id in participants:
                if agent_id in agent_moves:
                    moves = agent_moves[agent_id]
                    # Calculate cumulative path
                    path = []
                    current_pos = par_solver_input['start_positions'].get(agent_id, (0, 0))
                    path.append(current_pos)
                    
                    for di, dj in moves:
                        # ActorMove uses di/dj naming where di=row_increment, dj=col_increment
                        # In RL coordinate system: x=col, y=row, so dj->x, di->y
                        new_pos = (current_pos[0] + dj, current_pos[1] + di)
                        path.append(new_pos)
                        current_pos = new_pos
                    
                    path_str = " → ".join([f"({x}, {y})" for x, y in path])
                    # self.logger.info(f"   Agent {agent_id}: {path_str}")
                    
                    # Store trajectory in par_details
                    par_details['solution']['trajectories'][str(agent_id)] = {
                        'path': path,
                        'moves': moves,
                        'path_length': len(path)
                    }
                else:
                    # self.logger.info(f"   Agent {agent_id}: No trajectory computed")
                    par_details['solution']['trajectories'][str(agent_id)] = {
                        'path': [],
                        'moves': [],
                        'path_length': 0
                    }
        
        # Attach meta if available on result
        try:
            meta = {}
            for key in ['runtime', 'steps', 'stats']:
                if hasattr(par_solution, key):
                    meta[key] = getattr(par_solution, key)
            # copy any extra meta captured in coordinator via par_solver_input
            extra = par_solver_input.get('solution_meta') if isinstance(par_solver_input, dict) else None
            if extra:
                meta.update(extra)
            if meta:
                par_details['solution']['meta'] = meta
        except Exception:
            pass

        # Store in episode data
        if 'par_solver_details' not in self.episode_data:
            self.episode_data['par_solver_details'] = []
        self.episode_data['par_solver_details'].append(par_details)

        # Generate per-initialize PAR debug figure (static) under current run log dir
        try:
            init = par_details.get('initialization', {})
            starts_grid = init.get('start_positions', {})
            goals_grid = init.get('goal_positions', {})
            bounds = init.get('workspace_bounds', {})
            res = float(init.get('grid_resolution', 0.2))

            min_x = float(bounds.get('min_x', 0.0))
            max_x = float(bounds.get('max_x', 0.0))
            min_y = float(bounds.get('min_y', 0.0))
            max_y = float(bounds.get('max_y', 0.0))

            # Reconstruct continuous trajectories from agents_moves if available
            cont_paths = {}
            if par_solution and hasattr(par_solution, 'agents_moves') and isinstance(starts_grid, dict):
                # Build cumulative grid paths per agent id
                init_pos = {}
                for k, gp in starts_grid.items():
                    try:
                        aid = int(k)
                    except Exception:
                        aid = k
                    if isinstance(gp, (list, tuple)) and len(gp) == 2:
                        init_pos[aid] = (int(gp[0]), int(gp[1]))
                # Accumulate
                paths_grid = {aid: [pos] for aid, pos in init_pos.items()}
                for mv in getattr(par_solution, 'agents_moves', []) or []:
                    aid = getattr(mv, 'id', None)
                    di = getattr(mv, 'di', getattr(mv, 'dx', 0))  # row_increment (y direction)
                    dj = getattr(mv, 'dj', getattr(mv, 'dy', 0))  # col_increment (x direction)
                    if aid in paths_grid and paths_grid[aid]:
                        gx, gy = paths_grid[aid][-1]
                        # Fix coordinate system: di=row_increment (y), dj=col_increment (x)
                        paths_grid[aid].append((gx + dj, gy + di))
                # Convert to continuous (cell centers)
                for aid, gpath in paths_grid.items():
                    cont = [(min_x + (p[0] + 0.5) * res, min_y + (p[1] + 0.5) * res) for p in gpath]
                    cont_paths[aid] = cont

            # Convert starts/goals to continuous
            starts_cont = {}
            goals_cont = {}
            if isinstance(starts_grid, dict):
                for k, gp in starts_grid.items():
                    try:
                        aid = int(k)
                    except Exception:
                        aid = k
                    if isinstance(gp, (list, tuple)) and len(gp) == 2:
                        starts_cont[aid] = (min_x + (gp[0] + 0.5) * res, min_y + (gp[1] + 0.5) * res)
            if isinstance(goals_grid, dict):
                for k, gp in goals_grid.items():
                    try:
                        aid = int(k)
                    except Exception:
                        aid = k
                    if isinstance(gp, (list, tuple)) and len(gp) == 2:
                        goals_cont[aid] = (min_x + (gp[0] + 0.5) * res, min_y + (gp[1] + 0.5) * res)

            # Plot figure
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_aspect('equal')
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_title(f"PAR Init ep{self.stats['episode']} step{self.stats['step']}")
            # grid ticks per grid resolution
            if res > 0:
                ax.set_xticks(np.arange(min_x, max_x + 1e-9, res))
                ax.set_yticks(np.arange(min_y, max_y + 1e-9, res))
                ax.grid(True, linestyle='--', alpha=0.35)
            # If we have access to sub_map grid, overlay obstacle raster as a light gray image
            try:
                sub_map = par_details.get('sub_map', None)
                # Fallback: use current function arguments (par_solver_input)
                if sub_map is None and isinstance(par_solver_input, dict):
                    sub_map = par_solver_input.get('sub_map', None)
                grid = None
                if sub_map is not None and hasattr(sub_map, 'grid'):
                    grid = np.array(sub_map.grid, dtype=float)
                elif 'obstacle_grid' in par_details:
                    grid = np.array(par_details.get('obstacle_grid'))
                if grid is not None and grid.size > 0:
                    # grid is indexed [row(y), col(x)] with 1 for obstacle
                    # extent maps cell edges to continuous coordinates
                    extent = [min_x, max_x, min_y, max_y]
                    ax.imshow(grid, cmap='Greys', alpha=0.25, origin='lower', extent=extent, interpolation='nearest')
            except Exception:
                pass

            # submap rectangle and corner annotations
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='black', linewidth=1.0, linestyle=':')
            ax.add_patch(rect)
            ax.text(min_x, min_y, f"({min_x:.2f},{min_y:.2f})", fontsize=8)
            ax.text(max_x, max_y, f"({max_x:.2f},{max_y:.2f})", fontsize=8, ha='right', va='bottom')

            # choose colors
            agents = sorted(set(list(starts_cont.keys()) + list(goals_cont.keys()) + list(cont_paths.keys())))
            colors = plt.cm.get_cmap('tab10', max(10, len(agents)))
            color_map = {aid: colors(i % 10) for i, aid in enumerate(agents)}

            # draw goals
            for aid, (gx, gy) in goals_cont.items():
                c = color_map.get(aid, 'gray')
                ax.plot([gx], [gy], 'x', color=c, markersize=7, alpha=0.9)
                ax.text(gx, gy, f"g{aid}", fontsize=7, color=c, va='bottom', ha='left')
            # draw starts and paths
            for aid in agents:
                c = color_map.get(aid, 'gray')
                if aid in starts_cont:
                    sx, sy = starts_cont[aid]
                    ax.plot([sx], [sy], 'o', color=c, markersize=6, alpha=0.9)
                    ax.text(sx, sy, f"s{aid}", fontsize=7, color=c, va='top', ha='right')
                path = cont_paths.get(aid, [])
                if path and len(path) > 1:
                    xs = [p[0] for p in path]
                    ys = [p[1] for p in path]
                    ax.plot(xs, ys, '-', color=c, linewidth=1.5, alpha=0.8)
                elif aid in starts_cont and aid in goals_cont:
                    sx, sy = starts_cont[aid]
                    gx, gy = goals_cont[aid]
                    ax.plot([sx, gx], [sy, gy], linestyle='--', color=c, linewidth=1.0, alpha=0.5)

            # save file
            out_dir = self.log_dir / "par_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            outfile = out_dir / f"par_init_ep{self.stats['episode']:03d}_step{self.stats['step']:03d}.png"
            fig.savefig(str(outfile), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Also dump a JSON alongside the image for trajectory and diagnostics inspection
            try:
                jsonfile = out_dir / f"par_init_ep{self.stats['episode']:03d}_step{self.stats['step']:03d}.json"
                # Avoid non-serializable fields like sub_map objects
                serializable = {
                    'step': par_details.get('step'),
                    'initialization': par_details.get('initialization', {}),
                    'solution': par_details.get('solution', {}),
                    'obstacle_occupancy': par_details.get('initialization', {}).get('obstacle_occupancy')
                }
                with open(jsonfile, 'w') as jf:
                    json.dump(self._convert_to_serializable(serializable), jf, indent=2)
            except Exception as _:
                pass
        except Exception as e:
            # Do not break logging pipeline if visualization fails
            self.logger.debug(f"PAR VIS ERROR: {e}")
    
    def log_par_final_positions(self, agent_final_positions: Dict[int, List[float]]):
        """Log final agent positions after PAR execution completion."""
        # self.logger.info(f"🏁 PAR FINAL POSITIONS: Agent positions after PAR execution completion")
        for agent_id, position in agent_final_positions.items():
            # self.logger.info(f"   Agent {agent_id}: final_position={position}")
            pass
        
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


def get_deadlock_logger(log_dir: str = None, log_level: str = "DEBUG") -> DeadlockLogger:
    """Get or create global deadlock logger instance."""
    global _deadlock_logger
    if _deadlock_logger is None:
        _deadlock_logger = DeadlockLogger(log_dir, log_level)
    return _deadlock_logger


def reset_deadlock_logger():
    """Reset global deadlock logger instance."""
    global _deadlock_logger
    _deadlock_logger = None
