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
