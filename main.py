#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Celebrimbor - 机器人可视化编程平台
主入口文件
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from bin.core.config_manager import ConfigManager
from bin.ui import MainWindow
from models import get_model
from bin.core.logger import log_info, log_success, log_error, log_debug, log_warning
from utils.logger import setup_logger  # 这个保留，用于设置文件日志


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("Celebrimbor 启动中...")
    logger.info("=" * 60)
    
    # 加载配置
    config = ConfigManager()
    logger.info("配置文件加载完成")
    
    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 创建主窗口
    window = MainWindow(config)
    
    # 加载默认机器人模型
    default_robot = config.get('SIMULATION', 'default_robot', fallback='go2')
    logger.info(f"加载默认机器人: {default_robot}")
    
    try:
        model = get_model('unitree')
        if model:
            robot_instance = model(default_robot)
            window.set_robot_model(robot_instance)
            logger.info(f"✅ Unitree模型加载成功")
        else:
            logger.warning("⚠️ 未找到Unitree模型，使用模拟模式")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        logger.warning("⚠️ 继续使用模拟模式")
    
    # 显示窗口
    window.show()
    logger.info("主窗口已显示")
    
    # 运行应用
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
