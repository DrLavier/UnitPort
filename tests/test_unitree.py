#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unitree模型测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import get_model


def test_unitree_model_creation():
    """测试Unitree模型创建"""
    UnitreeModel = get_model('unitree')
    assert UnitreeModel is not None, "无法获取Unitree模型类"
    
    model = UnitreeModel('go2')
    assert model is not None, "无法创建Unitree模型实例"
    assert model.robot_type == 'go2', "机器人类型不正确"
    
    print("✅ Unitree模型创建测试通过")


def test_available_actions():
    """测试可用动作列表"""
    UnitreeModel = get_model('unitree')
    model = UnitreeModel('go2')
    
    actions = model.get_available_actions()
    assert len(actions) > 0, "没有可用动作"
    assert 'lift_right_leg' in actions, "缺少抬右腿动作"
    
    print(f"✅ 可用动作测试通过，共 {len(actions)} 个动作")
    print(f"   动作列表: {actions}")


def test_model_initialization():
    """测试模型初始化"""
    UnitreeModel = get_model('unitree')
    model = UnitreeModel('go2')
    
    success = model.initialize()
    assert success, "模型初始化失败"
    
    print("✅ 模型初始化测试通过")


if __name__ == '__main__':
    print("=" * 60)
    print("运行Unitree模型测试")
    print("=" * 60)
    
    try:
        test_unitree_model_creation()
        test_available_actions()
        test_model_initialization()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
