"""
Compatibility module.

Remote API agents were removed from the application. Use
``core.local_models.LocalModelManager`` for local text, voice and video models.
"""

ModelChoice = str


def build_agents(*args, **kwargs):
    raise RuntimeError(
        "Remote API agents have been removed. "
        "Load local models through core.local_models.LocalModelManager instead."
    )

