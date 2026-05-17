# Unified observation frame published on /unitport/robot_state.
# Consumed by host-side PolicyRunner (via ROS2EnvContext) at policy rate.

std_msgs/Header header
sensor_msgs/JointState joint_state
sensor_msgs/Imu imu
sensor_msgs/BatteryState battery

unitport_msgs/SensorEntry[] extra_sensors

string kinematic_class
string controller_type
string transport_mode
string session_id
