#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
节点执行引擎
负责执行图形化编程节点
"""

from typing import List, Dict, Any, Optional
from bin.core.logger import log_info, log_error, log_debug, log_warning

# 移除: from utils.logger import get_logger
# logger = get_logger()


class NodeExecutor:
    """节点执行引擎"""
    
    def __init__(self):
        """初始化节点执行引擎"""
        self.nodes = []
        self.connections = []
        self.execution_order = []
    
    def add_node(self, node_id: str, node_type: str, node_data: Dict[str, Any]):
        """
        添加节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            node_data: 节点数据
        """
        node = {
            'id': node_id,
            'type': node_type,
            'data': node_data,
            'inputs': [],
            'outputs': []
        }
        self.nodes.append(node)
        log_debug(f"添加节点: {node_id} ({node_type})")
    
    def add_connection(self, from_node: str, from_port: str,
                      to_node: str, to_port: str):
        """
        添加连接
        
        Args:
            from_node: 源节点ID
            from_port: 源端口
            to_node: 目标节点ID
            to_port: 目标端口
        """
        connection = {
            'from': {'node': from_node, 'port': from_port},
            'to': {'node': to_node, 'port': to_port}
        }
        self.connections.append(connection)
        log_debug(f"添加连接: {from_node}.{from_port} -> {to_node}.{to_port}")
    
    def build_execution_order(self) -> bool:
        """
        构建执行顺序（拓扑排序）
        
        Returns:
            是否构建成功
        """
        # 构建依赖图
        graph = {node['id']: [] for node in self.nodes}
        in_degree = {node['id']: 0 for node in self.nodes}
        
        for conn in self.connections:
            from_id = conn['from']['node']
            to_id = conn['to']['node']
            
            if from_id in graph and to_id in graph:
                graph[from_id].append(to_id)
                in_degree[to_id] += 1
        
        # 拓扑排序
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否存在循环
        if len(result) != len(self.nodes):
            log_error("节点图中存在循环依赖")
            return False
        
        self.execution_order = result
        log_info(f"执行顺序: {' -> '.join(result)}")
        return True
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行节点图
        
        Args:
            context: 执行上下文
        
        Returns:
            执行结果
        """
        if not self.execution_order:
            if not self.build_execution_order():
                raise RuntimeError("无法构建执行顺序")
        
        context = context or {}
        results = {}
        
        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if not node:
                log_warning(f"未找到节点: {node_id}")
                continue
            
            log_info(f"执行节点: {node_id} ({node['type']})")
            
            # 收集输入
            inputs = self._collect_inputs(node_id, results)
            
            # 执行节点
            try:
                output = self._execute_node(node, inputs, context)
                results[node_id] = output
                log_debug(f"节点 {node_id} 输出: {output}")
            except Exception as e:
                log_error(f"节点 {node_id} 执行失败: {e}")
                results[node_id] = {'error': str(e)}
        
        return results
    
    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """查找节点"""
        for node in self.nodes:
            if node['id'] == node_id:
                return node
        return None
    
    def _collect_inputs(self, node_id: str, 
                       results: Dict[str, Any]) -> Dict[str, Any]:
        """收集节点输入"""
        inputs = {}
        
        for conn in self.connections:
            if conn['to']['node'] == node_id:
                from_id = conn['from']['node']
                from_port = conn['from']['port']
                to_port = conn['to']['port']
                
                if from_id in results:
                    inputs[to_port] = results[from_id].get(from_port)
        
        return inputs
    
    def _execute_node(self, node: Dict[str, Any], 
                     inputs: Dict[str, Any],
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个节点
        
        Args:
            node: 节点信息
            inputs: 输入数据
            context: 执行上下文
        
        Returns:
            节点输出
        """
        # 这里应该根据节点类型调用相应的执行逻辑
        # 当前返回简单的模拟输出
        return {
            'status': 'success',
            'output': f"Executed {node['type']}",
            'data': node['data']
        }
    
    def clear(self):
        """清空所有节点和连接"""
        self.nodes.clear()
        self.connections.clear()
        self.execution_order.clear()
        log_info("清空节点执行器")
    
    def to_code(self) -> str:
        """
        将节点图转换为Python代码
        
        Returns:
            生成的代码
        """
        if not self.execution_order:
            self.build_execution_order()
        
        code_lines = [
            "# 自动生成的代码",
            "# Generated by Celebrimbor",
            "",
            "def execute_workflow():",
        ]
        
        for node_id in self.execution_order:
            node = self._find_node(node_id)
            if node:
                code_lines.append(f"    # {node['type']}: {node_id}")
                code_lines.append(f"    # TODO: 实现 {node['type']} 逻辑")
                code_lines.append("")
        
        code_lines.append("if __name__ == '__main__':")
        code_lines.append("    execute_workflow()")
        
        return "\n".join(code_lines)