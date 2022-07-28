from setuptools import setup, find_namespace_packages

setup(
    name='fastmri-intervention',
    version='1.6',
    packages=find_namespace_packages(include=["intervention", "intervention.*"]),
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
        'jsonschema~=4.6',
        'gcapi~=0.7',
        'picai-prep @ git+https://github.com/DIAGNijmegen/picai_prep@refactor',
        'click'
    ],
    entry_points={
        'console_scripts': [
            'fastmri-intervention = intervention.__main__:cli',
        ]
    }
)
