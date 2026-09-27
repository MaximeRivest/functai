"""``python -m functai verify <saved program folder>``."""

import sys

from .saved import main

raise SystemExit(main(sys.argv[1:]))
