from setuptools import setup

setup(
    name='saltswap',
    author='Gregory A. Ross',
    author_email='gregory.ross@choderalab.org',
    version='0.1dev',
    url='https://github.com/choderalab/saltswap',
    packages=['saltswap', 'saltswap.tests'],
    license='MIT',
    long_description=open('README.md').read(),
    platforms=[
            'Linux',
            'Mac OS-X',
            'Unix'],
    zip_safe=False,
    install_requires=[
        'simtk.openmm',
        'numpy',
        ],
)
