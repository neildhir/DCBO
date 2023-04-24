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

with open("README.md", "r") as fh:
    long_description = fh.read()

# enforce >Python3 for all versions of pip/setuptools
assert sys.version_info >= (3,), "This package requires Python 3."

requirements = [
    "emukit",
    "GPy",
    "graphviz",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "paramz",
    "pygraphviz",
    "scikit_learn",
    "scipy",
    "seaborn",
    "tqdm",
]

setup(
    name="DCBO",
    description="Toolkit for causal decision making under uncertainty.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neildhir/DCBO",
    packages=find_packages(exclude=["test*"]),
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3",
    license="General Public License",
)
