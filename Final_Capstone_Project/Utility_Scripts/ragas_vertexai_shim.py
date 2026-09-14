#################################################################################
# 
# Massachusetts Institute of Technology - xPRO
# RAG and Context Engineering: Designing and Building Production-Grade AI Systems
# 
# Student: Miguel Herrera
# Section: B
# Date and Time: 2026-09-14 11:00:00 -07:00
#
# Description: Compatibility shim that registers a stub for the
#              langchain_community.chat_models.vertexai module ragas imports
#              unconditionally (upstream issue explodinggradients/ragas#2995),
#              even though this project never uses VertexAI.
#
#################################################################################

"""Call ensure_ragas_vertexai_shim() before importing ragas."""
from __future__ import annotations

import importlib
import sys
import types

_MODULE_NAME = "langchain_community.chat_models.vertexai"
_PARENT_NAME = "langchain_community.chat_models"


def ensure_ragas_vertexai_shim() -> None:
    """Register a stub module only if the real one is missing; no-op once ragas is fixed upstream."""
    if _MODULE_NAME in sys.modules:
        return
    try:
        importlib.import_module(_MODULE_NAME)
        return
    except ModuleNotFoundError:
        pass

    stub_module = types.ModuleType(_MODULE_NAME)

    class ChatVertexAI:  # pragma: no cover - inert placeholder, never instantiated here
        """Placeholder so ragas can import a symbol this project never actually uses."""

    stub_module.ChatVertexAI = ChatVertexAI
    sys.modules[_MODULE_NAME] = stub_module

    parent_module = sys.modules.get(_PARENT_NAME)
    if parent_module is not None:
        parent_module.vertexai = stub_module
