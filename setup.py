import os

from setuptools import find_packages, setup

package_name = 'evoenv'
base_path = os.path.abspath(os.path.dirname(__file__))
version_path = os.path.join(base_path, package_name, '__version__.py')
version = {}
with open(version_path) as f:
	exec(f.read(), version)

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
	name=package_name,
	version=version['__version__'],
	packages=find_packages(exclude=["tests"]),
	author='Alex McAvoy',
	author_email='alexmcavoy@gmail.com',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/alexmcavoy/evoenv'
)
