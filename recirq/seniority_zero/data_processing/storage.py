# Copyright 2023 Google
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

"""Generic storage class for experiments."""

import os
import pickle
from abc import abstractmethod


class Experiment:
    """Generic data container for an experiment.

    ##### See TemplateExperiment below for how to subclass #####

    Effectively just a dictionary with load and save capability, and a special
    entry called 'keyword_args' that is treated on the same level for access,
    but stored separately
    """

    def __init__(self, **kwargs):
        """Initialize with any data that you want to set straight away."""
        self._data = self.default_data()
        for key in kwargs:
            self._data[key] = kwargs[key]

    def save(self, folder: str):
        """Save data stored in this experiment to file."""
        self._presave_processing()
        if folder[-1] == '/':
            folder = folder[:-1]
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        all_filenames = self.filenames
        for key in all_filenames:
            filename = all_filenames[key]
            data = self[key]
            if data is None:
                continue
            with open(f'{folder}/{filename}', 'wb') as outfile:
                pickle.dump(data, outfile)
        self._postsave_processing()

    def load(self, folder: str):
        """Load an experiment from a saved file."""
        if folder[-1] == '/':
            folder = folder[:-1]
        all_filenames = self.filenames
        for key in all_filenames:
            if key == 'keyword_args':
                continue  # Do this later
            filename = all_filenames[key]
            fpath = f'{folder}/{filename}'
            if os.path.exists(fpath) is False:
                continue
            with open(f'{folder}/{filename}', 'rb') as infile:
                self._data[key] = pickle.load(infile)
        # Treat keyword args specially
        filename = all_filenames['keyword_args']
        fpath = f'{folder}/{filename}'
        if os.path.exists(fpath) is True:
            with open(f'{folder}/{filename}', 'rb') as infile:
                keyword_args = pickle.load(infile)
            for key in keyword_args:
                self._data['keyword_args'][key] = keyword_args[key]
        self._postload_processing()

    # @classmethod
    @abstractmethod
    def default_data(self):
        """The data that is stored in this class by default."""
        raise NotImplementedError

    def keys_dont_save(self):
        """Properties of this class to not save"""
        return []

    def _presave_processing(self):
        """Function to call to do any processing before data saving"""

    def _postsave_processing(self):
        """Function to call to do any processing after data saving"""

    def _postload_processing(self):
        """Function to call to do any processing after data loading"""

    @property
    def filenames(self):
        """A filename for each piece of stored data that will be saved."""
        return {key: key + '.pkl' for key in self._data if key not in self.keys_dont_save()}

    # Boilerplate below to make this class behave mostly like a dictionary

    def __getitem__(self, key):
        """Override the standard dictionary getter."""
        if key in self._data:
            return self._data[key]
        elif key in self._data['keyword_args']:
            return self._data['keyword_args'][key]
        else:
            raise KeyError(f"Key '{key}'' not found in dictionary.")

    def __setitem__(self, key, item):
        """Override the standard dictionary setter."""
        if key in self._data:
            if key == 'keyword_args':
                raise KeyError('I cant set that.')
            self._data[key] = item
        elif key in self._data['keyword_args']:
            self._data['keyword_args'][key] = item
        else:
            raise KeyError(f"Key '{key}' isnt a parameter I understand.")

    def __repr__(self):
        """Inherit from standard dictionary boilerplate."""
        return repr(self._data)

    def __len__(self):
        """Inherit from standard dictionary boilerplate."""
        return len(self._data)

    def has_key(self, k):
        """Inherit from standard dictionary boilerplate."""
        return k in self._data or k in self._data['keyword_args']

    def keys(self):
        """Inherit from standard dictionary boilerplate."""
        return list(self._data.keys()) + list(self._data['keyword_args'].keys())

    def values(self):
        """Inherit from standard dictionary boilerplate."""
        return self._data.values()

    def items(self):
        """Inherit from standard dictionary boilerplate."""
        return self._data.items()

    def __cmp__(self, other):
        """Inherit from standard dictionary boilerplate."""
        return self.__cmp__(other)

    def __contains__(self, item):
        """Inherit from standard dictionary boilerplate."""
        return item in self._data or item in self._data['keyword_args']


class TemplateExperiment(Experiment):
    """Copy this and fill in to subclass Experiment"""

    @classmethod
    def default_data(self):
        """The data that can be stored in this experiment"""
        return {
            # Put data objects here (preferably everything
            # should be json-serializeable. If not, we can't save it, add it to
            # the list returned by 'keys_dont_save')
            'keyword_args': {
                # Put keywords / single parameters here, they will be
                # bundled together when this data is saved.
            }
        }

    @property
    def keys_dont_save(self):
        """List of keys in this experiment that won't be saved."""
        return []

    # Use these functions if needed
    def presave_processing(self):
        """Function to call to do any processing before data saving"""

    def postsave_processing(self):
        """Function to call to do any processing after data saving"""

    def postload_processing(self):
        """Function to call to do any processing after data loading"""
