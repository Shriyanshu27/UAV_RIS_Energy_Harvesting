from setuptools import setup, find_packages

setup(
    name='gym_foo',
    version='0.0.1',
    description='Hybrid RL UAV-RIS environment with PPO and DQN',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'gym==0.21.0',
        'numpy'
    ]
)
