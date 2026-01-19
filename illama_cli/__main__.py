"""Entry point for illama CLI.

Supports both:
- Running as a package: python -m illama_cli
- PyInstaller standalone binary
"""

import sys
import os

# Add the parent directory to path for PyInstaller compatibility
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    bundle_dir = os.path.dirname(sys.executable)
    sys.path.insert(0, bundle_dir)
    from illama_cli.cli import cli
else:
    # Running as normal Python module
    from .cli import cli


if __name__ == "__main__":
    cli()
