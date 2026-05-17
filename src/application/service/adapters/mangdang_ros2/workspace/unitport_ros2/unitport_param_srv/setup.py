from setuptools import setup

package_name = "unitport_param_srv"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="UnitPort",
    maintainer_email="noreply@unitport.dev",
    description="UnitPort parameter service.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "param_srv_node = unitport_param_srv.param_srv_node:main",
        ],
    },
)
