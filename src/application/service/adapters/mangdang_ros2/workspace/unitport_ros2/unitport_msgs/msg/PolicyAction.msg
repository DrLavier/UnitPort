# Policy → bridge action envelope. Fields are mutually exclusive: the
# consumer picks based on `space`.
#
#   space = "velocity_2d"    → linear/angular fields valid
#   space = "joint_position" → joint_positions valid
#   space = "joint_effort"   → joint_efforts valid
#   space = "stop"           → publish zero on active controller

std_msgs/Header header
string space
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
float64[] joint_positions
float64[] joint_efforts
