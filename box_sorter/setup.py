from setuptools import setup
import os
from glob import glob

pkg = 'box_sorter'
setup(
    name=pkg, version='0.1.0', packages=[pkg],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + pkg]),
        ('share/' + pkg, ['package.xml']),
        (os.path.join('share', pkg, 'launch'), glob('launch/*.py')),
        (os.path.join('share', pkg, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', pkg, 'config'), glob('config/*.yaml')),
        (os.path.join('share', pkg, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'box_spawner = box_sorter.box_spawner:main',
        'sorter_brain = box_sorter.sorter_brain:main',
        'navigator = box_sorter.navigator:main',
        'rviz_sim_node = box_sorter.rviz_sim_node:main',
    ]},
)
