from setuptools import setup


setup(
    name="pytorch_rl2_mdp_lstm",
    py_modules=["rl2"],
    version="2.0.0",
    description="A Pytorch implementation of RL^2.",
    author="Lucas D. Lingle",
    install_requires=[
        'mpi4py==3.0.3',
        'torch==1.8.1'
    ]
)