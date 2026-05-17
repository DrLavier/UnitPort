from setuptools import setup

package_name = "unitport_state_pub"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="UnitPort",
    maintainer_email="noreply@unitport.dev",
    description="Unified robot state publisher.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generic_state_pub_node = unitport_state_pub.generic_state_pub_node:main",
        ],
    },
)
