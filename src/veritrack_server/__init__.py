"""VeriTrack Stage 2 -- central ingestion, storage and hotlist matching.

Smart India Hackathon PS 26127 (Bharat Electronics Limited).

This package is the trust boundary of the platform. Stage 1 runs unattended on
roadside poles; everything it sends arrives here as untrusted input, is
validated, routed down the appropriate DPDP Act 2023 track, and persisted.

Typical use::

    from veritrack_server.config import Settings
    from veritrack_server.gateway import create_app

    app = create_app(Settings())

Or from the command line::

    uvicorn veritrack_server.gateway:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "config",
    "schemas",
    "crypto",
    "bloom",
    "db",
    "gateway",
]
