"""``python -m functai.bake supervise <run folder>``: a training run's own process."""

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "supervise":
        from .running import supervise
        supervise(argv[1])
        return 0
    if len(argv) == 2 and argv[0] == "worker":
        from .trainers.here_worker import main as worker
        return worker(argv[1])
    print("usage: python -m functai.bake supervise <run folder>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
