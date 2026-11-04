"""FunctAI Studio: see what a pipeline does, judge it, find the weak step, and
share how it works.

Your code stays the only source of truth. Studio reads it (``functai.check``),
runs it on examples, and keeps everything in plain files in the project:

    club.py                       the program (yours)
    examples/club_inbox.csv       examples: one row per input, plain CSV
    .functai/calls/               every run and judgement (FunctAI's call log)
    .functai/studio/              Studio's own: settings, what-if runs, trials

The command line (``functai studio --help``) is made for coding agents and
people alike; ``functai studio open`` starts the dashboard. Nothing here
changes FunctAI's contract: runs are ordinary call records whose ``caller``
says which example they answered, and judgements are ordinary ratings.
"""

from .project import Project, find_project, DEFAULT_LM  # noqa: F401
