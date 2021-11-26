# This program is free software; you can redistribute it and/or modify it under the terms of the 
# GNU General Public License as published by the Free Software Foundation; either version 3 of the License, 
# or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program. 
# If not, see http://www.gnu.org/licenses/.
# ==============================================================================

from setuptools import setup, find_packages
import sys

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req:
    requires = req.read().split("\n")

# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), 'This package requires Python 3.'

setup(
    name="DCBO",
    description='Toolkit for causal decision making under uncertainty.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neildhir/DCBO',
    packages=find_packages(exclude=['test*']),
    include_package_data=True,
    install_requires=requires,
    python_requires='>=3',
    license='General Public License',
    classifiers=(
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    )
)