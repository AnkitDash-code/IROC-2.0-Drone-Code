from setuptools import find_packages, setup

package_name = 'drone_vision_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # You might also want to add these lines if you have launch or config files in your package
        # ('share/' + package_name + '/launch', glob(os.path.join('launch', '*launch.[pxy][yem]*'))),
        # ('share/' + package_name + '/config', glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krishna',
    maintainer_email='krish.gollamudi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # This makes 'drone_vision_controller' command run 'vision_controller_node.py'
            'drone_vision_controller = drone_vision_control.vision_controller_node:main',
            # This makes 'corner_detection' command run 'corner_detection.py'
            'corner_detection = drone_vision_control.corner_detection:main',
        ],
    },
)
