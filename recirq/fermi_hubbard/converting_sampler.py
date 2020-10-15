# Copyright 2020 Google
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

from typing import Union, Callable, Iterable, List

import cirq
import pandas as pd


class ConvertingSampler(cirq.Sampler):
    """Sampler delegate which converts circuit before sampling.
    """

    def __init__(self,
                 sampler: cirq.Sampler,
                 convert_func: Union[
                     Callable[[cirq.Circuit], cirq.Circuit],
                     Iterable[Callable[[cirq.Circuit], cirq.Circuit]]]
                 ) -> None:
        """Initializes the sampler with conversion.

        Args:
            sampler: Delegate sampler where all invocations are forwarded to.
            convert_func: Either a function that converts the circuit, or list
                of the converting functions which are applied one by one before
                delegating to the target sampler.
        """
        self._sampler = sampler
        if isinstance(convert_func, Iterable):
            self._converters = convert_func
        else:
            self._converters = convert_func,

    def _convert(self, program: cirq.Circuit) -> cirq.Circuit:
        for converter in self._converters:
            program = converter(program)
        return program

    def run(
            self,
            program: cirq.Circuit,
            param_resolver: cirq.ParamResolverOrSimilarType = None,
            repetitions: int = 1,
    ) -> cirq.Result:
        program = self._convert(program)
        return self._sampler.run(program,
                                 param_resolver,
                                 repetitions)

    def sample(
            self,
            program: cirq.Circuit,
            *,
            repetitions: int = 1,
            params: cirq.Sweepable = None,
    ) -> pd.DataFrame:
        program = self._convert(program)
        return self._sampler.sample(program,
                                    repetitions=repetitions,
                                    params=params)

    def run_sweep(
            self,
            program: cirq.Circuit,
            params: cirq.Sweepable,
            repetitions: int = 1,
    ) -> List[cirq.Result]:
        program = self._convert(program)
        return self._sampler.run_sweep(program, params, repetitions)

    async def run_async(self, program: cirq.Circuit, *,
                       repetitions: int) -> cirq.Result:
        program = self._convert(program)
        return await self._sampler.run_async(program,
                                             repetitions=repetitions)

    async def run_sweep_async(
            self,
            program: cirq.Circuit,
            params: cirq.Sweepable,
            repetitions: int = 1,
    ) -> List[cirq.Result]:
        program = self._convert(program)
        return await self._sampler.run_sweep_async(program, params, repetitions)