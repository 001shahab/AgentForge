#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for AgentForge.

I'm including this for compatibility with older pip versions and tools that
don't fully support pyproject.toml yet. Poetry handles the actual build,
but this shim ensures broader compatibility.

Author: Prof. Shahab Anbarjafari
"""

from setuptools import setup

# Poetry handles everything, but we need this for editable installs
# and some older tooling that still looks for setup.py
if __name__ == "__main__":
    setup()

