# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

__version__ = ''
exec(open('recirq/_version.py').read())
assert __version__, 'Version string cannot be empty'

setup(name='recirq',
      version=__version__,
      url='http://github.com/quantumlib/cirq',
      author='The Cirq Developers',
      author_email='cirq@googlegroups.com',
      python_requires='>=3.6.0',
      install_requires=[
          'cirq',
      ],
      license='Apache 2',
      description="",
      long_description=open('README.md', encoding='utf-8').read(),
      packages=find_packages(),
      )
