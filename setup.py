from setuptools import setup

setup(
    name='prep',
    version='0.1',
    packages=['prep'],
    license='MIT',
    author='C.R. Noordman',
    author_email='stan.noordman@radboudumc.nl',
    description='',
    install_requires=[
        'python-box~=6.0',
        'shapely~=1.8',
        'numpy~=1.22',
        'numpy-quaternion~=2022.4',
        'tqdm~=4.64',
        'SimpleITK~=2.1',
        'gcapi~=0.7',
        'picai-prep @ git+https://github.com/snorthman/picai_prep.git',
        'click'
    ]
)
