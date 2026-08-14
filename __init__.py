"""H3 Motion Context - Timeline.

Visual video-editor timeline for MiniMax H3, powered natively by ComfyUI core
guides. No monkey-patches, fully compatible with all custom nodes.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
