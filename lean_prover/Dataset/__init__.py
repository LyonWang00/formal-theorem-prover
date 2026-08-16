"""Canonical raw and Pantograph-verified dataset materialization utilities.

The CLI module is intentionally not imported here.  Keeping package import
side-effect free avoids loading the Pantograph stack when callers only inspect
the dataset directory or run the module with ``python -m``.
"""
