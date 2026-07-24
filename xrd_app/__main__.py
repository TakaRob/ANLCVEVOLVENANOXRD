"""Run xrd-app as ``python -m xrd_app`` — a PATH-free entry point.

Identical to the ``xrd-app`` console script (``xrd_app.cli:main``), but invoked
through the interpreter. This lets the app run without putting ``~/.local/bin``
on PATH, which on shared beamline machines can shadow other programs' binaries.

    python -m xrd_app --help
    python -m xrd_app gui
"""

from .cli import main

if __name__ == "__main__":
    main()
