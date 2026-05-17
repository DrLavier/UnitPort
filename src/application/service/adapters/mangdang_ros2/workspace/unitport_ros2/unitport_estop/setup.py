from setuptools import setup

package_name = "unitport_estop"

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
    description="E-stop watchdog.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "estop_node = unitport_estop.estop_node:main",
        ],
    },
)
