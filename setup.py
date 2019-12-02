from setuptools import setup, find_packages

setup(
    name='NEAT',
    version='3.0.0',
    author='Damian Schmidt',
    author_email='damian.schmidt97@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy', 'pygame', 'gym-retro', 'matplotlib', 'graphviz'
    ],
)
