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

from sphinx.ext.napoleon import Config, GoogleDocstring
import inspect
import re


class GoogleDocstringToMarkdown(GoogleDocstring):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markdown_lines = []

    def _parse_attributes_section(self, section):
        lines = ['#### Attributes']
        for _name, _type, _desc in self._consume_fields():
            desc = ' '.join(_desc)
            lines += [f' - `{_name}`: {desc}']

        lines += ['']
        return lines

    def _parse_see_also_section(self, section):
        lines = self._consume_to_next_section()
        lines = [line.strip() for line in lines]
        return ['#### See Also', ' '.join(lines), '']


def display_markdown_docstring(cls):
    config = Config()
    gds = GoogleDocstringToMarkdown(inspect.cleandoc(cls.__doc__),
                                    config=config, what='class')
    gds_lines = [f'### {cls.__name__}'] + gds.lines()
    gds_lines = [re.sub(r':py:func:`(\w+)`', r'`\1`', line)
                 for line in gds_lines]

    from IPython.display import Markdown
    return Markdown('\n'.join(gds_lines))
