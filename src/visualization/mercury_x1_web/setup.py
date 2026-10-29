from glob import glob

from setuptools import setup

package_name = "mercury_x1_web"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/web", glob("web/*")),
        (f"share/{package_name}/meshes", glob("meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Konstantin Severov",
    description="Mercury X1 Rerun bridge and web control page",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "rerun_bridge = mercury_x1_web.rerun_bridge:main",
        "web_server = mercury_x1_web.web_server:main",
    ]},
)
