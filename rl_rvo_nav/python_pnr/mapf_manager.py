from __future__ import annotations

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from .mapf_session import MAPFSession


class MAPFManager:
    """
    高层协调器：管理死锁组的 MAPF 会话（原子开启、执行、结束/回滚）。

    最小对外接口：
    - try_start(groups, agent_states, env_adapter, waypoints_dict, time_step)
    - step_execute(current_positions, dt)
    - is_active(agent_id)
    - next_target(agent_id)
    - cancel_session(session_id) / cancel_all()
    """

    def __init__(self) -> None:
        self.active_sessions: Dict[int, MAPFSession] = {}
        self.agent_to_session: Dict[int, int] = {}
        self._next_session_id: int = 1

    def try_start(
        self,
        groups: List[Set[int]],
        agent_states: Dict[str, Dict[int, np.ndarray]],
        env_adapter: Any,
        waypoints_dict: Dict[int, List[np.ndarray]],
        time_step: int,
    ) -> List[int]:
        """为每个死锁组尝试创建并启动一个 MAPF 会话。

        返回成功启动的 session_id 列表。
        """
        started_sessions: List[int] = []
        assigned_in_this_round: Set[int] = set()
        positions: Dict[int, np.ndarray] = agent_states.get("positions", {})
        # 扩张半径（米）：将近邻纳入同一会话，避免单机会话无效
        expand_radius: float = 5.0

        # 合并与现有 session 冲突的组，或跳过已在 MAPF 的 agent
        for group in groups:
            # 排除已在其它会话中的 agent
            group = {aid for aid in group if aid not in self.agent_to_session}
            if not group:
                continue
            # 合并本轮已分配的，避免重复
            group = {aid for aid in group if aid not in assigned_in_this_round}
            if not group:
                continue

            # 扩张：将距离阈值内的agent也纳入
            expanded = set(group)
            for aid in list(group):
                pa = positions.get(aid)
                if pa is None:
                    continue
                for bid, pb in positions.items():
                    if bid in expanded or bid in self.agent_to_session or bid in assigned_in_this_round:
                        continue
                    if np.linalg.norm(pa[:2] - pb[:2]) <= expand_radius:
                        expanded.add(bid)
            group = expanded

            session_id = self._next_session_id
            self._next_session_id += 1

            session = MAPFSession(
                session_id=session_id,
                group_agent_ids=group,
                env_adapter=env_adapter,
                waypoints_dict=waypoints_dict,
            )

            success = session.prepare_and_solve(agent_states, time_step)
            if success:
                # 原子提交：会话生效
                self.active_sessions[session_id] = session
                for aid in group:
                    self.agent_to_session[aid] = session_id
                    assigned_in_this_round.add(aid)
                started_sessions.append(session_id)
            else:
                # 失败自动回滚，跳过
                continue

        return started_sessions

    def step_execute(self, current_positions: Dict[int, np.ndarray], dt: float) -> None:
        """推进所有活跃会话的执行阶段。"""
        finished_sessions: List[int] = []
        for sid, session in list(self.active_sessions.items()):
            session.step(current_positions, dt)
            if session.is_finished:
                finished_sessions.append(sid)

        # 清理已完成会话
        for sid in finished_sessions:
            # 打印会话结束时的位姿信息（便于分析退出点是否安全）
            try:
                agents = sorted(list(self.active_sessions[sid].group_agent_ids)) if sid in self.active_sessions else []
                if agents:
                    info = {aid: current_positions.get(aid) for aid in agents}
                    print(f"[MAPFManager] session {sid} finished; final positions: {info}")
            except Exception:
                pass
            session = self.active_sessions.pop(sid, None)
            if session is None:
                continue
            for aid in session.group_agent_ids:
                self.agent_to_session.pop(aid, None)

    def is_active(self, agent_id: int) -> bool:
        return agent_id in self.agent_to_session

    def next_target(self, agent_id: int) -> Optional[np.ndarray]:
        sid = self.agent_to_session.get(agent_id)
        if sid is None:
            return None
        session = self.active_sessions.get(sid)
        if session is None:
            return None
        return session.next_target(agent_id)

    def cancel_session(self, session_id: int) -> None:
        session = self.active_sessions.pop(session_id, None)
        if session is None:
            return
        session.cancel()
        for aid in session.group_agent_ids:
            self.agent_to_session.pop(aid, None)

    def cancel_all(self) -> None:
        for sid, session in list(self.active_sessions.items()):
            session.cancel()
        self.active_sessions.clear()
        self.agent_to_session.clear()


