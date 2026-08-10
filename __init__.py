"""H3 Motion Context.

Clip chaining for MiniMax H3: pin the tail of the previous clip (picture
and sound) so the next clip genuinely continues it.

Runtime patches are installed lazily. Importing this extension does not modify
ComfyUI core classes. The first execution of H3 Motion Context or H3 Custom
Keyframes installs and self-tests both MiniMax H3 compatibility patches.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
