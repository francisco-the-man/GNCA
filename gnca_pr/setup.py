from setuptools import setup, find_packages

setup(
    name='gnca',
    version='0.1.0',
    description='Implementation of "Learning Graph Cellular Automata" in PyTorch Geometric',
    author='Avery Louis, Titus Lawrence Parker',
    author_email='averylou@stanford.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'scipy',
    ],
    extras_require={
        'examples': ['matplotlib', 'pygsp'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)