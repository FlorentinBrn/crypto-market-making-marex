"""Interfaces utilisateur : dashboard terminal (Rich) et dashboard web (Dash).

- `console`  : dashboard terminal Rich (embarqué dans le process du feed)
- `dash_app` : dashboard web Dash (process séparé, lit les CSV)
- `config`   : Settings centralisés utilisés partout
"""
from .config import Settings

__all__ = ["Settings"]
