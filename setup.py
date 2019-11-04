from setuptools import setup, find_packages

setup(
    name='NEAT',
    version='1.1.0',
    author='Damian Schmidt',
    author_email='damian.schmidt97@gmail.com',
    packages=find_packages(),
    install_requires=[
        'graphics.py', 'numpy'
    ],
    tests_require=[
        'parameterized'
    ],
)