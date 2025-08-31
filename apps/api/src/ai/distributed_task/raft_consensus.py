"""Raft分布式共识算法实现"""

import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import logging

from .models import (
    ConsensusState,
    RaftLogEntry,
    VoteRequest,
    VoteResponse,
    AppendEntriesRequest,
    AppendEntriesResponse,
)


class RaftConsensusEngine:
    """Raft共识引擎"""
    
    def __init__(
        self, 
        node_id: str, 
        cluster_nodes: List[str],
        message_bus=None,
        election_timeout_min: float = 2.0,
        election_timeout_max: float = 4.0,
        heartbeat_interval: float = 0.5
    ):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.message_bus = message_bus
        self.logger = logging.getLogger(__name__)
        
        # Raft状态
        self.state = ConsensusState.FOLLOWER
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        
        # 日志
        self.log: List[RaftLogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader专用状态
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # 选举配置
        self.election_timeout_min = election_timeout_min
        self.election_timeout_max = election_timeout_max
        self.heartbeat_interval = heartbeat_interval
        self.election_timer: Optional[asyncio.Task] = None
        self.heartbeat_timer: Optional[asyncio.Task] = None
        
        # 投票追踪
        self.votes_received: Set[str] = set()
        self.current_election_id: Optional[str] = None
        
        # 快照
        self.snapshot_index = 0
        self.snapshot_term = 0
        self.snapshot_data: Dict[str, Any] = {}
        
        # 回调函数
        self.apply_callback = None
        self.state_change_callback = None
        
        # 启动选举定时器
        self._reset_election_timer()
    
    async def start(self):
        """启动共识引擎"""
        self.logger.info(f"Node {self.node_id} starting Raft consensus engine")
        
        # 初始化为Follower
        await self._become_follower(None)
        
        # 启动消息处理循环
        if self.message_bus:
            asyncio.create_task(self._message_handler())
    
    async def stop(self):
        """停止共识引擎"""
        self.logger.info(f"Node {self.node_id} stopping Raft consensus engine")
        
        try:
            # 取消定时器
            if self.election_timer:
                self.election_timer.cancel()
            if self.heartbeat_timer:
                self.heartbeat_timer.cancel()
            
            # 等待定时器完成
            if self.election_timer and not self.election_timer.done():
                try:
                    await asyncio.wait_for(self.election_timer, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
            if self.heartbeat_timer and not self.heartbeat_timer.done():
                try:
                    await asyncio.wait_for(self.heartbeat_timer, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
        except Exception as e:
            self.logger.error(f"Error stopping Raft consensus engine: {e}")
        finally:
            self.state = ConsensusState.FOLLOWER
            self.logger.info(f"Node {self.node_id} stopped Raft consensus engine")
    
    async def append_entry(self, command: Dict[str, Any]) -> bool:
        """添加日志条目（客户端请求）"""
        
        if self.state != ConsensusState.LEADER:
            self.logger.warning(f"Node {self.node_id} is not leader, cannot append entry")
            return False
        
        # 创建日志条目
        entry = RaftLogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            timestamp=datetime.now(),
            command_type=command.get("action", "unknown"),
            command_data=command,
            client_id=command.get("client_id", "unknown"),
            sequence_number=command.get("sequence_number", 0)
        )
        
        # 添加到本地日志
        self.log.append(entry)
        
        # 向followers发送AppendEntries
        await self._replicate_entries()
        
        return True
    
    async def _become_follower(self, leader_id: Optional[str]):
        """转换为Follower状态"""
        
        self.logger.info(f"Node {self.node_id} becoming follower (term {self.current_term})")
        
        self.state = ConsensusState.FOLLOWER
        self.leader_id = leader_id
        self.voted_for = None
        
        # 重置选举定时器
        self._reset_election_timer()
        
        # 取消心跳定时器
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
            self.heartbeat_timer = None
        
        # 触发状态变更回调
        if self.state_change_callback:
            await self.state_change_callback(ConsensusState.FOLLOWER)
    
    async def _become_candidate(self):
        """转换为Candidate状态"""
        
        self.logger.info(f"Node {self.node_id} becoming candidate (term {self.current_term + 1})")
        
        self.state = ConsensusState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.leader_id = None
        self.votes_received = {self.node_id}
        self.current_election_id = str(uuid.uuid4())
        
        # 重置选举定时器
        self._reset_election_timer()
        
        # 向所有节点请求投票
        await self._request_votes()
        
        # 触发状态变更回调
        if self.state_change_callback:
            await self.state_change_callback(ConsensusState.CANDIDATE)
    
    async def _become_leader(self):
        """转换为Leader状态"""
        
        self.logger.info(f"Node {self.node_id} becoming leader (term {self.current_term})")
        
        self.state = ConsensusState.LEADER
        self.leader_id = self.node_id
        
        # 初始化Leader状态
        for node in self.cluster_nodes:
            if node != self.node_id:
                self.next_index[node] = len(self.log) + 1
                self.match_index[node] = 0
        
        # 取消选举定时器
        if self.election_timer:
            self.election_timer.cancel()
            self.election_timer = None
        
        # 启动心跳定时器
        self._start_heartbeat_timer()
        
        # 立即发送心跳
        await self._send_heartbeats()
        
        # 触发状态变更回调
        if self.state_change_callback:
            await self.state_change_callback(ConsensusState.LEADER)
    
    async def _request_votes(self):
        """请求投票"""
        
        last_log_index = len(self.log)
        last_log_term = self.log[-1].term if self.log else 0
        
        vote_request = VoteRequest(
            term=self.current_term,
            candidate_id=self.node_id,
            last_log_index=last_log_index,
            last_log_term=last_log_term,
            election_id=self.current_election_id
        )
        
        # 向所有节点发送投票请求
        for node in self.cluster_nodes:
            if node != self.node_id:
                await self._send_vote_request(node, vote_request)
    
    async def _send_vote_request(self, node: str, request: VoteRequest):
        """发送投票请求"""
        
        if self.message_bus:
            await self.message_bus.publish(
                f"raft.vote_request.{node}",
                {
                    "from": self.node_id,
                    "to": node,
                    "request": {
                        "term": request.term,
                        "candidate_id": request.candidate_id,
                        "last_log_index": request.last_log_index,
                        "last_log_term": request.last_log_term,
                        "election_id": request.election_id
                    }
                }
            )
    
    async def handle_vote_request(self, request: VoteRequest) -> VoteResponse:
        """处理投票请求"""
        
        vote_granted = False
        reason = None
        
        # 如果请求的term小于当前term，拒绝
        if request.term < self.current_term:
            reason = "Outdated term"
        # 如果请求的term大于当前term，更新term并转为follower
        elif request.term > self.current_term:
            self.current_term = request.term
            await self._become_follower(None)
        
        # 检查是否已经投票
        if self.voted_for is None or self.voted_for == request.candidate_id:
            # 检查日志是否至少和候选人一样新
            last_log_index = len(self.log)
            last_log_term = self.log[-1].term if self.log else 0
            
            if (request.last_log_term > last_log_term or
                (request.last_log_term == last_log_term and 
                 request.last_log_index >= last_log_index)):
                vote_granted = True
                self.voted_for = request.candidate_id
                self._reset_election_timer()
            else:
                reason = "Log not up to date"
        else:
            reason = f"Already voted for {self.voted_for}"
        
        return VoteResponse(
            term=self.current_term,
            vote_granted=vote_granted,
            voter_id=self.node_id,
            reason=reason
        )
    
    async def handle_vote_response(self, response: VoteResponse):
        """处理投票响应"""
        
        # 如果响应的term大于当前term，转为follower
        if response.term > self.current_term:
            self.current_term = response.term
            await self._become_follower(None)
            return
        
        # 如果不是candidate状态，忽略
        if self.state != ConsensusState.CANDIDATE:
            return
        
        # 如果响应的term不匹配，忽略
        if response.term != self.current_term:
            return
        
        # 记录投票
        if response.vote_granted:
            self.votes_received.add(response.voter_id)
            
            # 检查是否获得多数票
            if len(self.votes_received) > len(self.cluster_nodes) / 2:
                await self._become_leader()
    
    async def _send_heartbeats(self):
        """发送心跳"""
        
        if self.state != ConsensusState.LEADER:
            return
        
        # 向所有follower发送心跳
        for node in self.cluster_nodes:
            if node != self.node_id:
                await self._send_append_entries(node, heartbeat=True)
    
    async def _send_append_entries(self, node: str, heartbeat: bool = False):
        """发送AppendEntries请求"""
        
        if node not in self.next_index:
            self.next_index[node] = 1
        
        prev_log_index = self.next_index[node] - 1
        prev_log_term = 0
        
        if prev_log_index > 0 and prev_log_index <= len(self.log):
            prev_log_term = self.log[prev_log_index - 1].term
        
        # 获取要发送的日志条目
        entries = []
        if not heartbeat and self.next_index[node] <= len(self.log):
            entries = self.log[self.next_index[node] - 1:]
        
        request = AppendEntriesRequest(
            term=self.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=self.commit_index,
            heartbeat=heartbeat
        )
        
        if self.message_bus:
            await self.message_bus.publish(
                f"raft.append_entries.{node}",
                {
                    "from": self.node_id,
                    "to": node,
                    "request": {
                        "term": request.term,
                        "leader_id": request.leader_id,
                        "prev_log_index": request.prev_log_index,
                        "prev_log_term": request.prev_log_term,
                        "entries": [entry._calculate_hash() for entry in entries],
                        "leader_commit": request.leader_commit,
                        "heartbeat": request.heartbeat
                    }
                }
            )
    
    async def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """处理AppendEntries请求"""
        
        success = False
        match_index = 0
        conflict_index = None
        reason = None
        
        # 如果请求的term小于当前term，拒绝
        if request.term < self.current_term:
            reason = "Outdated term"
        else:
            # 如果请求的term大于等于当前term，更新term并转为follower
            if request.term > self.current_term:
                self.current_term = request.term
                self.voted_for = None
            
            await self._become_follower(request.leader_id)
            
            # 重置选举定时器
            self._reset_election_timer()
            
            # 检查日志一致性
            if request.prev_log_index > 0:
                if (request.prev_log_index > len(self.log) or
                    self.log[request.prev_log_index - 1].term != request.prev_log_term):
                    # 日志不一致
                    reason = "Log inconsistency"
                    conflict_index = min(request.prev_log_index, len(self.log))
                else:
                    success = True
            else:
                success = True
            
            if success:
                # 删除冲突的日志条目
                if request.entries:
                    self.log = self.log[:request.prev_log_index]
                    self.log.extend(request.entries)
                
                match_index = request.prev_log_index + len(request.entries)
                
                # 更新commit index
                if request.leader_commit > self.commit_index:
                    self.commit_index = min(request.leader_commit, len(self.log))
                    await self._apply_committed_entries()
        
        return AppendEntriesResponse(
            term=self.current_term,
            success=success,
            match_index=match_index,
            follower_id=self.node_id,
            conflict_index=conflict_index,
            reason=reason
        )
    
    async def handle_append_entries_response(self, response: AppendEntriesResponse):
        """处理AppendEntries响应"""
        
        # 如果响应的term大于当前term，转为follower
        if response.term > self.current_term:
            self.current_term = response.term
            await self._become_follower(None)
            return
        
        # 如果不是leader状态，忽略
        if self.state != ConsensusState.LEADER:
            return
        
        # 如果响应的term不匹配，忽略
        if response.term != self.current_term:
            return
        
        if response.success:
            # 更新next_index和match_index
            self.next_index[response.follower_id] = response.match_index + 1
            self.match_index[response.follower_id] = response.match_index
            
            # 检查是否可以提交新的日志条目
            await self._check_commit_index()
        else:
            # 如果失败，减少next_index并重试
            if response.conflict_index:
                self.next_index[response.follower_id] = response.conflict_index
            else:
                self.next_index[response.follower_id] = max(1, self.next_index[response.follower_id] - 1)
            
            # 重试发送
            await self._send_append_entries(response.follower_id)
    
    async def _check_commit_index(self):
        """检查并更新commit index"""
        
        if self.state != ConsensusState.LEADER:
            return
        
        # 找到大多数节点都已复制的最高索引
        for index in range(self.commit_index + 1, len(self.log) + 1):
            if self.log[index - 1].term == self.current_term:
                replicated_count = 1  # Leader自己
                
                for node in self.cluster_nodes:
                    if node != self.node_id and self.match_index.get(node, 0) >= index:
                        replicated_count += 1
                
                if replicated_count > len(self.cluster_nodes) / 2:
                    self.commit_index = index
        
        # 应用已提交的日志条目
        await self._apply_committed_entries()
    
    async def _apply_committed_entries(self):
        """应用已提交的日志条目"""
        
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]
            
            # 调用应用回调
            if self.apply_callback:
                await self.apply_callback(entry)
            
            self.logger.debug(f"Node {self.node_id} applied log entry {self.last_applied}")
    
    async def _replicate_entries(self):
        """复制日志条目到所有follower"""
        
        if self.state != ConsensusState.LEADER:
            return
        
        for node in self.cluster_nodes:
            if node != self.node_id:
                await self._send_append_entries(node)
    
    def _reset_election_timer(self):
        """重置选举定时器"""
        
        if self.election_timer:
            self.election_timer.cancel()
        
        # 随机选举超时
        timeout = random.uniform(self.election_timeout_min, self.election_timeout_max)
        self.election_timer = asyncio.create_task(self._election_timeout(timeout))
    
    async def _election_timeout(self, timeout: float):
        """选举超时处理"""
        
        await asyncio.sleep(timeout)
        
        if self.state == ConsensusState.FOLLOWER:
            self.logger.info(f"Node {self.node_id} election timeout, starting election")
            await self._become_candidate()
    
    def _start_heartbeat_timer(self):
        """启动心跳定时器"""
        
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        
        while self.state == ConsensusState.LEADER:
            await self._send_heartbeats()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _message_handler(self):
        """消息处理循环"""
        
        if not self.message_bus:
            return
        
        # 订阅Raft消息
        await self.message_bus.subscribe(f"raft.vote_request.{self.node_id}", self._handle_vote_request_message)
        await self.message_bus.subscribe(f"raft.vote_response.{self.node_id}", self._handle_vote_response_message)
        await self.message_bus.subscribe(f"raft.append_entries.{self.node_id}", self._handle_append_entries_message)
        await self.message_bus.subscribe(f"raft.append_entries_response.{self.node_id}", self._handle_append_entries_response_message)
    
    async def _handle_vote_request_message(self, message: Dict[str, Any]):
        """处理投票请求消息"""
        
        request_data = message["request"]
        request = VoteRequest(
            term=request_data["term"],
            candidate_id=request_data["candidate_id"],
            last_log_index=request_data["last_log_index"],
            last_log_term=request_data["last_log_term"],
            election_id=request_data["election_id"]
        )
        
        response = await self.handle_vote_request(request)
        
        # 发送响应
        await self.message_bus.publish(
            f"raft.vote_response.{message['from']}",
            {
                "from": self.node_id,
                "to": message["from"],
                "response": {
                    "term": response.term,
                    "vote_granted": response.vote_granted,
                    "voter_id": response.voter_id,
                    "reason": response.reason
                }
            }
        )
    
    async def _handle_vote_response_message(self, message: Dict[str, Any]):
        """处理投票响应消息"""
        
        response_data = message["response"]
        response = VoteResponse(
            term=response_data["term"],
            vote_granted=response_data["vote_granted"],
            voter_id=response_data["voter_id"],
            reason=response_data.get("reason")
        )
        
        await self.handle_vote_response(response)
    
    async def _handle_append_entries_message(self, message: Dict[str, Any]):
        """处理AppendEntries消息"""
        
        # 实际实现中需要反序列化日志条目
        # 这里简化处理
        pass
    
    async def _handle_append_entries_response_message(self, message: Dict[str, Any]):
        """处理AppendEntries响应消息"""
        
        response_data = message["response"]
        response = AppendEntriesResponse(
            term=response_data["term"],
            success=response_data["success"],
            match_index=response_data["match_index"],
            follower_id=response_data["follower_id"],
            conflict_index=response_data.get("conflict_index"),
            reason=response_data.get("reason")
        )
        
        await self.handle_append_entries_response(response)
    
    async def create_snapshot(self) -> Dict[str, Any]:
        """创建快照"""
        
        snapshot = {
            "index": self.last_applied,
            "term": self.log[self.last_applied - 1].term if self.last_applied > 0 else 0,
            "data": self.snapshot_data.copy(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 压缩日志
        if self.last_applied > 0:
            self.log = self.log[self.last_applied:]
            self.snapshot_index = self.last_applied
            self.snapshot_term = snapshot["term"]
        
        self.logger.info(f"Node {self.node_id} created snapshot at index {snapshot['index']}")
        
        return snapshot
    
    async def load_snapshot(self, snapshot: Dict[str, Any]):
        """加载快照"""
        
        self.snapshot_index = snapshot["index"]
        self.snapshot_term = snapshot["term"]
        self.snapshot_data = snapshot["data"]
        self.last_applied = snapshot["index"]
        self.commit_index = snapshot["index"]
        
        # 清空日志
        self.log = []
        
        self.logger.info(f"Node {self.node_id} loaded snapshot at index {snapshot['index']}")