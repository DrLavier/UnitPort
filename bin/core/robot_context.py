#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Context - Global Robot State Management

This module provides a centralized way to manage the current robot type and model instance.
All action nodes should use this context to get the appropriate robot model based on
the user's selection in the UI.

Design Pattern:
    - Singleton-like global state for robot configuration
    - Factory method to create robot model based on brand/type
    - Automatic model routing based on robot_type

Usage:
    from bin.core.robot_context import RobotContext

    # Set robot type (called by UI when user selects robot)
    RobotContext.set_robot_type("go2")

    # Get current robot model (used by action nodes)
    robot = RobotContext.get_robot_model()
    if robot:
        robot.run_action("stand")
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from bin.core.logger import log_info, log_error, log_debug, log_warning
from design.service.service_registry import ServiceRegistry
from design.service.service_router import ServiceRouter
from design.service.adapters.unitree_sdk2.adapter import UnitreeAdapter

if TYPE_CHECKING:
    from models.base import BaseRobotModel


class RobotContext:
    """
    Global Robot Context Manager

    Manages the current robot type and provides the appropriate robot model
    instance based on the selected robot brand and type.

    Supported robot types and their brand mappings:
        - go2, a1, b1, b2, h1 -> unitree
        - (future) spot -> boston_dynamics
        - (future) anymal -> anybotics
    """

    # Global state
    _current_robot_type: str = "go2"
    _current_robot_model: Optional['BaseRobotModel'] = None
    _initialized: bool = False
    _service_registry: ServiceRegistry = ServiceRegistry()
    _service_router: ServiceRouter = ServiceRouter(_service_registry)

    # Robot type to brand mapping
    ROBOT_BRAND_MAP: Dict[str, str] = {
        # Unitree robots
        "go2": "unitree",
        "a1": "unitree",
        "b1": "unitree",
        "b2": "unitree",
        "h1": "unitree",
        # Add more brands here as needed
        # "spot": "boston_dynamics",
        # "anymal": "anybotics",
    }

    # Available robot types per brand
    BRAND_ROBOTS: Dict[str, list] = {
        "unitree": ["go2", "a1", "b1", "b2", "h1"],
        # "boston_dynamics": ["spot"],
        # "anybotics": ["anymal"],
    }

    BRAND_ADAPTER_MAP: Dict[str, str] = {
        "unitree": "unitree_sdk2",
    }

    @classmethod
    def set_robot_type(cls, robot_type: str) -> bool:
        """
        Set the current robot type and create appropriate model instance.

        Args:
            robot_type: Robot type identifier (e.g., "go2", "a1", "b1")

        Returns:
            True if successfully set, False otherwise
        """
        robot_type = robot_type.lower()

        # Check if robot type is supported
        if robot_type not in cls.ROBOT_BRAND_MAP:
            log_warning(f"Unknown robot type: {robot_type}, defaulting to 'go2'")
            robot_type = "go2"

        # If same type and already initialized, skip
        if robot_type == cls._current_robot_type and cls._current_robot_model is not None:
            log_debug(f"Robot type already set to: {robot_type}")
            return True

        cls._current_robot_type = robot_type
        cls._current_robot_model = None  # Clear old model
        cls._initialized = False

        log_info(f"Robot type set to: {robot_type} (brand: {cls.get_current_brand()})")
        return True

    @classmethod
    def get_robot_type(cls) -> str:
        """Get the current robot type."""
        return cls._current_robot_type

    @classmethod
    def get_current_brand(cls) -> str:
        """Get the brand of the current robot type."""
        return cls.ROBOT_BRAND_MAP.get(cls._current_robot_type, "unitree")

    @classmethod
    def get_robot_model(cls, force_reinit: bool = False) -> Optional['BaseRobotModel']:
        """
        Get the robot model instance for the current robot type.
        Creates the model lazily on first access.

        Args:
            force_reinit: If True, recreate the model even if already initialized

        Returns:
            Robot model instance, or None if creation fails
        """
        if cls._current_robot_model is not None and cls._initialized and not force_reinit:
            return cls._current_robot_model

        # Create model based on brand
        brand = cls.get_current_brand()
        robot_type = cls._current_robot_type

        try:
            adapter = cls._ensure_adapter(brand, robot_type, force_reinit=force_reinit)
            model = adapter.get_model() if adapter is not None else None
            if model:
                cls._current_robot_model = model
                cls._initialized = True
                log_info(f"Robot model created: {brand}/{robot_type}")
            return model
        except Exception as e:
            log_error(f"Failed to create robot model: {e}")
            return None

    @classmethod
    def _create_model_for_brand(cls, brand: str, robot_type: str) -> Optional['BaseRobotModel']:
        """
        Factory method to create robot model based on brand.

        This is the key routing mechanism - it imports and instantiates
        the correct model class from the appropriate brand module.

        Args:
            brand: Robot brand (e.g., "unitree")
            robot_type: Robot type within the brand (e.g., "go2")

        Returns:
            Robot model instance
        """
        if brand == "unitree":
            from models.unitree import UnitreeModel
            return UnitreeModel(robot_type)

        # Add more brands here
        # elif brand == "boston_dynamics":
        #     from models.boston_dynamics import BostonDynamicsModel
        #     return BostonDynamicsModel(robot_type)

        else:
            log_warning(f"Unknown brand: {brand}, falling back to unitree")
            from models.unitree import UnitreeModel
            return UnitreeModel(robot_type)

    @classmethod
    def _ensure_adapter(cls, brand: str, robot_type: str, force_reinit: bool = False):
        """Ensure adapter is registered and bound to current robot type."""
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand, "unitree_sdk2")
        adapter = cls._service_registry.get(adapter_name)

        if adapter is None:
            if adapter_name == "unitree_sdk2":
                adapter = UnitreeAdapter(robot_type)
            else:
                log_warning(f"Unknown adapter '{adapter_name}', falling back to unitree_sdk2")
                adapter = UnitreeAdapter(robot_type)
            cls._service_registry.register(adapter_name, adapter)

        try:
            adapter.connect(robot_type=robot_type, force_reinit=force_reinit)
        except Exception as e:
            log_error(f"Adapter connect failed ({adapter_name}): {e}")
            return None

        return adapter

    @classmethod
    def run_action(cls, action_name: str, **kwargs) -> bool:
        """
        Execute an action on the current robot model.

        This is a convenience method that action nodes can use directly.

        Args:
            action_name: Name of the action to execute
            **kwargs: Action parameters

        Returns:
            True if action executed successfully, False otherwise
        """
        brand = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand, "unitree_sdk2")
        adapter = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            log_error(f"Cannot execute action '{action_name}': Adapter unavailable")
            return False
        try:
            return bool(cls._service_router.run_action(adapter_name, action_name, **kwargs))
        except Exception as e:
            log_error(f"Action routing failed ({action_name}): {e}")
            robot = cls.get_robot_model()
            if robot is None:
                return False
            return bool(robot.run_action(action_name, **kwargs))

    @classmethod
    def get_sensor_data(cls) -> Dict[str, Any]:
        """
        Get sensor data from the current robot model.

        Returns:
            Sensor data dictionary
        """
        brand = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand, "unitree_sdk2")
        adapter = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            return {'error': 'No robot model available'}
        try:
            return cls._service_router.get_sensor_data(adapter_name)
        except Exception as e:
            log_error(f"Sensor routing failed: {e}")
            robot = cls.get_robot_model()
            if robot is None:
                return {'error': 'No robot model available'}
            return robot.get_sensor_data()

    @classmethod
    def stop(cls):
        """Stop the current robot."""
        brand = cls.get_current_brand()
        adapter_name = cls.BRAND_ADAPTER_MAP.get(brand, "unitree_sdk2")
        adapter = cls._ensure_adapter(brand, cls._current_robot_type)
        if adapter is None:
            return
        try:
            cls._service_router.stop(adapter_name)
        except Exception as e:
            log_error(f"Stop routing failed: {e}")
            robot = cls.get_robot_model()
            if robot:
                robot.stop()

    @classmethod
    def get_available_actions(cls) -> list:
        """Get list of available actions for current robot."""
        robot = cls.get_robot_model()
        if robot:
            return robot.get_available_actions()
        return []

    @classmethod
    def is_available(cls) -> bool:
        """Check if robot model is available and ready."""
        robot = cls.get_robot_model()
        return robot is not None and robot.is_available

    @classmethod
    def get_supported_robots(cls) -> Dict[str, list]:
        """Get all supported robots grouped by brand."""
        return cls.BRAND_ROBOTS.copy()

    @classmethod
    def reset(cls):
        """Reset the context to initial state."""
        cls._current_robot_model = None
        cls._initialized = False
        cls._service_registry = ServiceRegistry()
        cls._service_router = ServiceRouter(cls._service_registry)
        log_debug("Robot context reset")


# Convenience functions for direct import
def get_robot() -> Optional['BaseRobotModel']:
    """Convenience function to get current robot model."""
    return RobotContext.get_robot_model()


def run_action(action_name: str, **kwargs) -> bool:
    """Convenience function to run action on current robot."""
    return RobotContext.run_action(action_name, **kwargs)


def get_sensor_data() -> Dict[str, Any]:
    """Convenience function to get sensor data from current robot."""
    return RobotContext.get_sensor_data()
