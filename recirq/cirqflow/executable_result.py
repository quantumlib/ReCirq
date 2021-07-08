from dataclasses import dataclass
from typing import List, Union, Tuple
from collections import defaultdict


def _recurse_html_rg(rg, depth=0):
    if isinstance(rg, ResultGroup):
        s = ' ' * depth + f'<li>{rg}<ul>\n'

        for child in rg.results:
            s += _recurse_html_rg(child, depth=depth + 1)
        s += ' ' * depth + '</ul></li>\n'
        return s

    s = ' ' * depth + '<li>' + str(rg) + '</li>\n'
    return s


@dataclass(frozen=True)
class Result:
    result: str


@dataclass(frozen=True)
class ResultGroup:
    results: Tuple[Union['ResultGroup', Result], ...]

    def __repr__(self):
        return f'ResultGroup({hash(self)}) at {id(self)}'

    def __str__(self):
        return f'ResultGroup({hash(self)})'

    def _repr_html_(self):
        return '<ul>' + _recurse_html_rg(self) + '</ul>'


def retree_results(x, children):
    if isinstance(x, Result):
        assert False

    if all(isinstance(child, (Result, ResultGroup)) for child in children[x]):
        return ResultGroup(results=tuple(children[x]))

    rgs = [retree_results(child, children) for child in children[x]]
    return ResultGroup(results=tuple(rgs))


# TODO: move to runtime
def _execute_flattened(flat_list):
    children = defaultdict(set)

    for program, info, parents in flat_list:
        result = Result(
            result=str(program.uuid) + 'r'
        )

        # TODO: order
        children[parents[0]].add(result)

        for p1, p2 in zip(parents, parents[1:]):
            children[p2].add(p1)

    return children
