"""Where training happens. One small protocol, several places:

    "here"     this machine's GPUs (TRL, PEFT, Accelerate)
    "tinker"   Thinking Machines' training API
    "prime"    Prime Intellect's hosted SFT
    "export"   nothing runs: the examples and ready configs for TRL, Axolotl, Unsloth
    a Trainer  yours

A trainer says what it is missing (``check``), what a job would take
(``estimate``: seconds, dollars, and the settings it would train with), and
runs it (``supervise``, in the run's own process). ``choose`` picks a place
for ``where="auto"``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from ..examples import BakeError


class Stopped(Exception):
    """The run was asked to stop, and stopped (after saving a checkpoint)."""


@dataclasses.dataclass
class Job:
    """What a trainer is asked to estimate: the student, the data, the options."""
    student: Any                          # StudentInfo
    stats: Dict[str, Any]                 # Examples.stats() of the training split
    train_rows: int
    options: Dict[str, Any]               # the user's training options (lora, epochs, lr, ...)


@dataclasses.dataclass
class Estimate:
    """A place's answer for a job."""
    ok: bool
    problems: List[str] = dataclasses.field(default_factory=list)
    seconds: Optional[float] = None
    dollars: Optional[float] = None
    settings: Dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)
    summary: str = ""                     # one line for the plan

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class Trainer:
    """A place training happens (see the module docstring)."""
    name = "trainer"
    paid = False                          # a service that bills; never used by "auto" unless set up

    def set_up(self) -> bool:
        """Whether the user has set this place up (a key, a login). Places
        that are not set up are listed in a plan, never chosen by "auto"."""
        return True

    def offers(self, student: str) -> Optional[bool]:
        """Whether it trains ``student`` (None: cannot tell)."""
        return True

    def estimate(self, job: Job) -> Estimate:
        raise NotImplementedError

    def supervise(self, run) -> None:
        """Run the job to its end, in the run's own process (resumes from the
        run's last checkpoint)."""
        raise NotImplementedError

    def download(self, baked, path):
        raise BakeError(f"{self.name} keeps no weights elsewhere")

    def checkpoint_model(self, run, step: int):
        """A checkpoint of ``run`` as a Baked model."""
        raise BakeError(f"{self.name} keeps no checkpoints bake can load")


def trainer(name: Any) -> Trainer:
    if isinstance(name, Trainer):
        return name
    if name == "here":
        from .here import Here
        return Here()
    if name == "tinker":
        from .tinker import Tinker
        return Tinker()
    if name == "prime":
        from .prime import Prime
        return Prime()
    if name == "export":
        from .export import Export
        return Export()
    raise BakeError(f"where= is 'auto', 'here', 'tinker', 'prime', 'export', a list of them, or a Trainer; "
                    f"not {name!r}")


PLACES = ("here", "tinker", "prime", "export")

__all__ = ["Trainer", "Job", "Estimate", "Stopped", "trainer", "PLACES"]
