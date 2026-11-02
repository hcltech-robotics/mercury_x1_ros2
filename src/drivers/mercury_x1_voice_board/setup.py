from glob import glob

from setuptools import setup

package_name = "mercury_x1_voice_board"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/udev", glob("udev/*.rules")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Konstantin Severov",
    description="Mercury X1 head voice board driver (wake word, direction of arrival, beam/gain control)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={"console_scripts": [
        "voice_board_node = mercury_x1_voice_board.voice_board_node:main",
    ]},
)
