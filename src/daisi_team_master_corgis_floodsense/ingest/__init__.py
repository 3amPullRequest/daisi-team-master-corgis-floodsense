"""Ingestion: fetch NEA, PUB, Telegram, UHSLC and static layers, and land them raw in the UC landing volume.

`landing/<name>/...` is streamed by Auto Loader into `bronze.<name>` (see src/pipeline/utilities/bronze_sources.py).
"""
