from setuptools import setup

setup(
    name='FastMRIInterventionReconstruction',
    version='0.1',
    packages=['annotator'],
    url='',
    license='MIT',
    author='C.R. Noordman',
    author_email='stan.noordman@radboudumc.nl',
    description='',
    install_requires=[
        'python-box~=6.0',
        'shapely~=1.8',
        'numpy~=1.22',
        'numpy-quaternion~=2022.4',
        # 1.3.3 has a small bug for python3.10; fixed in live, waiting for proper release
        'p_tqdm',
        'SimpleITK~=2.1.1',
        'gcapi~=0.7',
        'picai-prep @ https://github.com/DIAGNijmegen/picai_prep/archive/refs/tags/v1.1.1.zip'
    ],
    dependency_links = [
        'git+https://github.com/swansonk14/p_tqdm.git/@master'
    ]
)
