from setuptools import setup, find_packages

setup(
    name='NEAT',
    version='2.2.0',
    author='Damian Schmidt',
    author_email='damian.schmidt97@gmail.com',
    packages=find_packages(),
    install_requires=[
        'graphics.py', 'numpy', 'pygame', 'gym-retro', 'matplotlib',
    ],
    tests_require=[
        'parameterized'
    ],
)