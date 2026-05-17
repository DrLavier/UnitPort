from setuptools import setup

package_name = "unitport_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.adapters",
        f"{package_name}.aux_handlers",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="UnitPort",
    maintainer_email="noreply@unitport.dev",
    description="UnitPort generic bridge node.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generic_bridge_node = unitport_bridge.generic_bridge_node:main",
        ],
    },
)
