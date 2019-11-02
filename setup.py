from setuptools import setup, find_packages

setup(
    name='NEAT_conventional',
    version='1.0.0',
    author='Damian Schmidt',
    author_email='damian.schmidt97@gmail.com',
    packages=find_packages(),
    install_requires=[
        'graphics.py'
    ],
    tests_require=[
        'parameterized'
    ],
)