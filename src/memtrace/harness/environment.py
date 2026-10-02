"""Safe loading of protected provider and proxy environment files.

The historical Codex benchmark launcher accepted shell-style ``export`` files
for credentials and transport settings.  MemTrace accepts the same small
assignment subset without executing the file, expanding variables, or
printing its values.  This keeps credentials outside configs and command-line
arguments while making the old launch contract reproducible.
"""
from __future__ import annotations

import os
import re
import shlex
import stat
from collections.abc import Iterable
from pathlib import Path


class EnvironmentFileError(ValueError):
    """A protected environment file is missing, unsafe, or malformed."""


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def read_environment_file(path: Path) -> dict[str, str]:
    """Read simple ``NAME=value`` assignments from a private file.

    Values may use shell quoting, but no shell is invoked and command or
    variable substitution is never performed.  Group/other permission bits
    are rejected so a provider key cannot accidentally be exposed to another
    local user.
    """

    selected = Path(path).expanduser().resolve()
    try:
        metadata = selected.stat()
    except OSError as exc:
        raise EnvironmentFileError(f"environment file is not readable: {selected}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise EnvironmentFileError(f"environment file is not a regular file: {selected}")
    if metadata.st_mode & 0o077:
        raise EnvironmentFileError(
            f"environment file must be private (mode 600 or stricter): {selected}"
        )
    try:
        lines = selected.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EnvironmentFileError(f"environment file cannot be read: {selected}") from exc

    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvironmentFileError(
                f"environment file line {line_number} is not a NAME=value assignment"
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if _NAME.fullmatch(name) is None:
            raise EnvironmentFileError(
                f"environment file line {line_number} has an invalid variable name"
            )
        try:
            tokens = shlex.split(raw_value, comments=False, posix=True)
        except ValueError as exc:
            raise EnvironmentFileError(
                f"environment file line {line_number} has invalid shell quoting"
            ) from exc
        if len(tokens) > 1:
            raise EnvironmentFileError(
                f"environment file line {line_number} contains multiple values"
            )
        value = tokens[0] if tokens else ""
        if "\x00" in value:
            raise EnvironmentFileError(f"environment file line {line_number} contains NUL")
        result[name] = value
    return result


def apply_environment_files(paths: Iterable[Path]) -> tuple[str, ...]:
    """Load private files into the current process and return changed names.

    Only variable names are returned.  Values never appear in receipts or
    command output.
    """

    changed: list[str] = []
    for path in paths:
        values = read_environment_file(path)
        os.environ.update(values)
        changed.extend(values)
    return tuple(dict.fromkeys(changed))
