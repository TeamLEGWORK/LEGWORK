import sys
import os
import nbsphinx
import re

# credit to https://github.com/rodluger/starry_process

# Add the CWD to the path
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# Hack `nbsphinx` to enable us to hide certain input cells in the
# jupyter notebooks. This works with nbsphinx==0.5.0
nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{% block input -%}",
    '{% block input -%}\n{%- if not "hide_input" in cell.metadata.tags %}',
)
nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{% endblock input %}", "{% endif %}\n{% endblock input %}"
)

# Hack `nbsphinx` to prevent fixed-height images, which look
# terrible when the window is resized!
nbsphinx.RST_TEMPLATE = re.sub(
    r"\{%- if height %\}.*?{% endif %}",
    "",
    nbsphinx.RST_TEMPLATE,
    flags=re.DOTALL,
)
