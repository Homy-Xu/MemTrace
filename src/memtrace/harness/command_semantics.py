from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass

_SHELLS = frozenset({"bash", "dash", "sh", "zsh", "ksh"})
_TEST_RUNNERS = frozenset({"pytest", "py.test", "tox", "nox", "jest", "vitest", "mocha", "ava", "tap", "testem"})
_TEST_SCRIPT_NAMES = frozenset(
    {"test", "test.sh", "tests", "tests.sh", "run-tests", "run-tests.sh", "run_tests.sh"}
)
_COMMAND_BOUNDARIES = frozenset({"&&", "||", ";", "|", "|&", "&"})
# A test runner invoked only to describe itself observes no behavior.  Treating
# ``pytest --version`` as a TEST_RESULT bound a Criterion to tooling
# troubleshooting in the 2.2.90 stress run and hid the real gap.
_INFORMATIONAL_RUNNER_FLAGS = frozenset(
    {
        "--version",
        "-V",
        "--help",
        "-h",
        "--collect-only",
        "--co",
        "--fixtures",
        "--fixtures-per-test",
        "--markers",
        "--setup-plan",
        "--setup-only",
        "--dry-run",
        # A snapshot-updating run rewrites the expectation instead of checking
        # it.  In an earlier SWE-Milestone run the model ran ``jest -u`` on suites
        # it had broken and then treated the green result as verification;
        # the official evaluator kept the original snapshots and scored those
        # suites as regressions.  Snapshot/expectation rewrites therefore never
        # count as a test observation.
        "-u",
        "--updateSnapshot",
        "--update-snapshots",
        "--update-snapshot",
        "--update",
        "--snapshot-update",
        "--accept",
        "--insta-accept",
        "-Pupdate-snapshots",
        "-Dsnapshot.update=true",
    }
)


# Process wrappers that run the command that follows them.  ``timeout 120
# python -m pytest`` is the form many AGENTS.md files prescribe; in the 2.2.92
# pressure run every test the model ran was wrapped this way and none of them
# was recognized as a test observation, so the Milestone could never be
# verified.  Each wrapper maps to the options that consume a separate value.
_COMMAND_WRAPPERS: Mapping[str, frozenset[str]] = {
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "ionice": frozenset({"-c", "-n", "-p", "--class", "--classdata"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "nohup": frozenset(),
    "exec": frozenset(),
    "unbuffer": frozenset(),
    "caffeinate": frozenset(),
    "xvfb-run": frozenset({"-s", "--server-args", "-n", "--server-num", "-f", "-e", "-p"}),
    "sudo": frozenset({"-u", "--user", "-g", "--group", "-p", "-C", "-h", "--host"}),
}
_TIMEOUT_DURATION = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")


def _skip_command_wrappers(segment: tuple[str, ...], offset: int) -> int:
    """Advance past leading process wrappers to the executable they launch."""

    while offset < len(segment):
        while offset < len(segment) and (
            segment[offset] == "!"
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[offset]) is not None
        ):
            offset += 1
        if offset >= len(segment):
            return offset
        wrapper = _basename(segment[offset])
        valued = _COMMAND_WRAPPERS.get(wrapper)
        if valued is None:
            return offset
        cursor = offset + 1
        while cursor < len(segment) and segment[cursor].startswith("-"):
            option = segment[cursor]
            if option == "--":
                cursor += 1
                break
            cursor += 1
            if option in valued and cursor < len(segment):
                cursor += 1
        if wrapper == "timeout":
            if cursor < len(segment) and _TIMEOUT_DURATION.match(segment[cursor]):
                cursor += 1
            else:
                # ``timeout`` without a duration is not a wrapper we understand.
                return offset
        if cursor >= len(segment):
            # A wrapper with nothing to launch is an ordinary command.
            return offset
        offset = cursor
    return offset


@dataclass(frozen=True, slots=True)
class CommandEvidenceSemantics:
    """One deterministic interpretation shared by Planning and execution."""

    is_test_observation: bool
    success_exit_status_reliable: bool


@dataclass(frozen=True, slots=True)
class LogicalCommandOutcome:
    """A selector-local result proved by one Provider command item."""

    success: bool
    exit_code: int
    basis: str


@dataclass(frozen=True, slots=True)
class _LocatedTestCommand:
    script: str
    tokens: tuple[str, ...]
    executable_index: int
    segment_end: int


def _basename(token: str) -> str:
    return token.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _unwrap_shell_script(command: str) -> str:
    try:
        outer = shlex.split(command)
    except ValueError:
        return command
    for index, token in enumerate(outer[:-2]):
        if _basename(token) in _SHELLS and outer[index + 1] in {"-c", "-lc"}:
            return outer[index + 2]
    return command


def _shell_tokens(script: str) -> tuple[str, ...]:
    try:
        lexer = shlex.shlex(script, posix=True, punctuation_chars="|&;")
        lexer.whitespace_split = True
        raw = tuple(lexer)
    except ValueError:
        return ()
    # With ``&`` configured as shell punctuation, shlex represents the common
    # file-descriptor redirect ``2>&1`` as (``2>``, ``&``, ``1``).  The middle
    # token is not a command boundary and must not split the test pipeline.
    normalized: list[str] = []
    index = 0
    while index < len(raw):
        if (
            index + 2 < len(raw)
            and raw[index].endswith((">", "<"))
            and raw[index + 1] == "&"
            and re.fullmatch(r"(?:\d+|-)", raw[index + 2]) is not None
        ):
            normalized.append("".join(raw[index : index + 3]))
            index += 3
            continue
        normalized.append(raw[index])
        index += 1
    return tuple(normalized)


def _python_failure_probe(source: str) -> bool:
    """Return whether Python source can prove a broken behavior by failing.

    A print-only inspection is useful diagnostic output, but it cannot satisfy
    a ``TEST_FAILURE`` contract because success and failure have the same exit
    status. Assertions, raises and explicit exits have an observable failure
    boundary and therefore can produce typed failure Evidence.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assert, ast.Raise)):
            return True
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in {"exit", "quit"}:
            return True
        if (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "sys"
            and function.attr == "exit"
        ):
            return True
    return False


def _locate_test_command(command: str) -> _LocatedTestCommand | None:
    """Locate an executable test or assertion probe, not a textual mention."""

    script = _unwrap_shell_script(command)
    tokens = _shell_tokens(script)
    if not tokens:
        return None

    starts = (
        0,
        *(index + 1 for index, token in enumerate(tokens) if token in _COMMAND_BOUNDARIES),
    )
    for start in starts:
        end = next(
            (
                index
                for index in range(start, len(tokens))
                if tokens[index] in _COMMAND_BOUNDARIES
            ),
            len(tokens),
        )
        segment = tokens[start:end]
        if not segment:
            continue
        offset = _skip_command_wrappers(segment, 0)
        if offset >= len(segment):
            continue
        executable = _basename(segment[offset])
        arguments = segment[offset + 1 :]
        if any(argument in _INFORMATIONAL_RUNNER_FLAGS for argument in arguments):
            continue

        if executable in _TEST_RUNNERS:
            return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable in {"mvn", "mvnw", "mvn.cmd", "mvnw.cmd"}:
            skip = any(re.fullmatch(r"-d(?:skiptests|maven\.test\.skip|skipits)(?:=true)?", a.lower()) for a in arguments)
            goals = {"test", "verify", "integration-test", "surefire:test", "failsafe:integration-test", "failsafe:verify"}
            if not skip and any(a in goals for a in arguments):
                return _LocatedTestCommand(script, tokens, start + offset, end)
            continue
        if executable in {"gradle", "gradlew", "gradle.bat", "gradlew.bat"}:
            if "-m" in arguments or "-x" in arguments or any(a.startswith("--exclude-task") for a in arguments):
                continue
            if any(a.rsplit(":", 1)[-1] in {"test", "integrationTest", "functionalTest"} for a in arguments):
                return _LocatedTestCommand(script, tokens, start + offset, end)
            continue
        if executable == "node" and "--test" in arguments:
            return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable in {"npx", "pnpm", "yarn", "bun"}:
            runner_args = arguments
            while runner_args and runner_args[0] in {"exec", "dlx", "-y", "--yes"}:
                runner_args = runner_args[1:]
            if runner_args and _basename(runner_args[0]) in _TEST_RUNNERS:
                return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable.startswith(("python", "pypy")):
            for index, token in enumerate(arguments):
                if token == "-c":
                    if index + 1 < len(arguments) and _python_failure_probe(arguments[index + 1]):
                        return _LocatedTestCommand(script, tokens, start + offset, end)
                    break
                if token == "-m" and index + 1 < len(arguments):
                    module = arguments[index + 1].casefold()
                    if module in {"pytest", "unittest"}:
                        return _LocatedTestCommand(script, tokens, start + offset, end)
                    break
                if token.startswith("-"):
                    continue
                normalized = token.casefold().replace("\\", "/")
                basename = _basename(token)
                if "/tests/" in f"/{normalized}" or basename.startswith(
                    ("test_", "run_tests", "run-tests")
                ):
                    return _LocatedTestCommand(script, tokens, start + offset, end)
                break
            continue
        if executable in {"cargo", "go"} and arguments[:1] == ("test",):
            if "--no-run" in arguments or "--list" in arguments or "-list" in arguments or any(a.startswith("-list=") for a in arguments):
                continue
            return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable in {"npm", "pnpm", "yarn", "bun"} and (
            arguments[:1] == ("test",)
            or len(arguments) >= 2
            and arguments[0] == "run"
            and arguments[1].casefold().startswith("test")
        ):
            return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable == "make" and any(
            argument.casefold().startswith("test") for argument in arguments
        ):
            return _LocatedTestCommand(script, tokens, start + offset, end)
        if executable in {"uv", "poetry", "pdm", "pipenv", "hatch", "coverage"}:
            for index, token in enumerate(arguments):
                candidate = _basename(token)
                if candidate in _TEST_RUNNERS:
                    return _LocatedTestCommand(script, tokens, start + offset + index + 1, end)
                if (
                    token == "-m"
                    and index + 1 < len(arguments)
                    and arguments[index + 1].casefold() in {"pytest", "unittest"}
                ):
                    return _LocatedTestCommand(script, tokens, start + offset + index + 1, end)
        normalized_executable = segment[offset].casefold().replace("\\", "/")
        if (
            "/tests/" in f"/{normalized_executable}"
            or executable in _TEST_SCRIPT_NAMES
            or executable.startswith(("test_", "run_tests", "run-tests"))
        ):
            return _LocatedTestCommand(script, tokens, start + offset, end)
    return None


def _python_probe_signature(located: _LocatedTestCommand) -> str | None:
    executable = _basename(located.tokens[located.executable_index])
    if not executable.startswith(("python", "pypy")):
        return None
    arguments = located.tokens[located.executable_index + 1 : located.segment_end]
    for index, token in enumerate(arguments[:-1]):
        if token != "-c":
            continue
        try:
            return ast.dump(ast.parse(arguments[index + 1]), include_attributes=False)
        except SyntaxError:
            return None
    return None


def _target_aliases(tokens: tuple[str, ...]) -> frozenset[str]:
    aliases: set[str] = set()
    for token in tokens:
        if not token or token.startswith("-") or token in _COMMAND_BOUNDARIES:
            continue
        normalized = token.casefold().replace("\\", "/")
        aliases.add(normalized)
        aliases.add(_basename(normalized))
        if "::" in normalized:
            aliases.update(part for part in normalized.split("::") if part)
    return frozenset(aliases)


def _test_address(located: _LocatedTestCommand) -> tuple[str, frozenset[str]]:
    executable = _basename(located.tokens[located.executable_index])
    arguments = located.tokens[located.executable_index + 1 : located.segment_end]
    if executable in _TEST_RUNNERS:
        return executable, _target_aliases(arguments)
    if executable.startswith(("python", "pypy")):
        for index, token in enumerate(arguments):
            if token == "-m" and index + 1 < len(arguments):
                module = arguments[index + 1].casefold()
                if module in {"pytest", "unittest"}:
                    return module, _target_aliases(arguments[index + 2 :])
            if token == "-c":
                return "python-probe", frozenset()
        return "python-test-file", _target_aliases(arguments)
    if executable in {"uv", "poetry", "pdm", "pipenv", "hatch", "coverage"}:
        for index, token in enumerate(arguments):
            candidate = _basename(token)
            if candidate in _TEST_RUNNERS:
                return candidate, _target_aliases(arguments[index + 1 :])
            if token == "-m" and index + 1 < len(arguments):
                module = arguments[index + 1].casefold()
                if module in {"pytest", "unittest"}:
                    return module, _target_aliases(arguments[index + 2 :])
    if executable in {"cargo", "go"}:
        return f"{executable}-test", _target_aliases(arguments[1:])
    if executable in {"mvn", "mvnw", "mvn.cmd", "mvnw.cmd"}:
        return "maven-test", _target_aliases(arguments)
    if executable in {"gradle", "gradlew", "gradle.bat", "gradlew.bat"}:
        return "gradle-test", _target_aliases(arguments)
    if executable in {"npm", "pnpm", "yarn", "bun", "make"}:
        return f"{executable}-test", _target_aliases(arguments)
    return "test-target", _target_aliases((located.tokens[located.executable_index], *arguments))


def _and_chain_segments(tokens: tuple[str, ...]) -> tuple[tuple[str, ...], ...] | None:
    """Split a pure ``&&`` chain into complete command segments."""

    boundaries = tuple(token for token in tokens if token in _COMMAND_BOUNDARIES)
    if any(token != "&&" for token in boundaries):
        return None
    segments: list[tuple[str, ...]] = []
    start = 0
    for index in range(len(tokens) + 1):
        if index < len(tokens) and tokens[index] != "&&":
            continue
        segment = tokens[start:index]
        if not segment:
            return None
        segments.append(segment)
        start = index + 1
    return tuple(segments)


def _is_success_implied_and_subchain(
    selector_tokens: tuple[str, ...],
    observed_tokens: tuple[str, ...],
) -> bool:
    """Return whether a successful compound command proves one exact subchain.

    A Provider commonly wraps a logical command with ``cd ... &&`` or batches
    a few independent observations in one shell item.  Treating the complete
    shell string as the address made exact selectors disappear behind this
    transport wrapper.  Conversely, arbitrary token-subsequence matching is
    unsound: in ``a ; b`` or ``a || b`` the final zero exit status does not
    prove that ``a`` succeeded.

    We therefore translate only complete command segments in a pure ``&&``
    selector to an exact, contiguous subchain of a pure ``&&`` Provider
    command.  When the enclosing item succeeds, shell semantics prove that
    every selected segment succeeded.  Pipelines, ``;``, ``||`` and background
    execution remain ineligible.
    """

    if not selector_tokens or not observed_tokens:
        return False
    selected = _and_chain_segments(selector_tokens)
    observed = _and_chain_segments(observed_tokens)
    if selected is None or observed is None or len(observed) < len(selected):
        return False
    normalized_selector = tuple(tuple(map(str.casefold, segment)) for segment in selected)
    normalized_observed = tuple(tuple(map(str.casefold, segment)) for segment in observed)
    width = len(normalized_selector)
    for start in range(len(normalized_observed) - width + 1):
        if normalized_observed[start : start + width] == normalized_selector:
            return True
    return False


def _strip_workspace_cd_transport(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Remove only the Harness-owned leading ``cd <cwd> &&`` address wrapper.

    Address translation and result truth are separate questions.  A failed
    wrapped command still addresses the declared selector; its exit status is
    evaluated later by the acceptance kernel.  General failed ``&&`` batches
    remain ineligible because their final status cannot identify which
    arbitrary subcommand ran.
    """

    segments = _and_chain_segments(tokens)
    if (
        segments is None
        or len(segments) < 2
        or len(segments[0]) != 2
        or segments[0][0].casefold() != "cd"
    ):
        return tokens
    flattened: list[str] = []
    for index, segment in enumerate(segments[1:]):
        if index:
            flattened.append("&&")
        flattened.extend(segment)
    return tuple(flattened)


def _command_segments(
    tokens: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], str | None], ...]:
    """Return complete top-level shell segments with their following operator."""

    segments: list[tuple[tuple[str, ...], str | None]] = []
    start = 0
    for index, token in enumerate(tokens):
        if token not in _COMMAND_BOUNDARIES:
            continue
        segment = tokens[start:index]
        if segment:
            segments.append((segment, token))
        start = index + 1
    tail = tokens[start:]
    if tail:
        segments.append((tail, None))
    return tuple(segments)


def _command_segment_index(tokens: tuple[str, ...], token_index: int) -> int | None:
    """Map one token offset to the corresponding non-empty shell segment."""

    segment_index = 0
    start = 0
    for index, token in enumerate(tokens):
        if token not in _COMMAND_BOUNDARIES:
            continue
        if start <= token_index < index:
            return segment_index
        if start < index:
            segment_index += 1
        start = index + 1
    if start <= token_index < len(tokens):
        return segment_index
    return None


def _echoed_previous_exit_code(
    segment: tuple[str, ...],
    output: str,
) -> int | None:
    """Read an explicit ``echo label=$?`` receipt for the prior segment.

    The non-empty label is intentional. A bare numeric output is too easy to
    confuse with ordinary program output and therefore is not an acceptance
    receipt.
    """

    if not segment or _basename(segment[0]) != "echo":
        return None
    arguments = list(segment[1:])
    while arguments and arguments[0] in {"-n", "-e", "-E"}:
        arguments.pop(0)
    template = " ".join(arguments)
    if template.count("$?") != 1:
        return None
    prefix, suffix = template.split("$?", 1)
    if not prefix and not suffix:
        return None
    pattern = re.compile(
        rf"(?m)^{re.escape(prefix)}(-?\d+){re.escape(suffix)}\s*$"
    )
    matches = pattern.findall(output)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _echoed_pipeline_exit_code(
    segment: tuple[str, ...],
    output: str,
    *,
    pipeline_index: int,
) -> int | None:
    """Read an explicit Bash ``PIPESTATUS[n]`` receipt for one producer.

    The Provider process exit code describes the trailing ``echo`` in a command
    such as ``pytest | tail; echo test_exit=${PIPESTATUS[0]}``, not pytest.
    The labelled value is nevertheless an exact, observable receipt for that
    pipeline element.  Accept only the matching index and a full output line;
    arbitrary success-looking test text remains inadmissible.
    """

    if not segment or _basename(segment[0]) != "echo":
        return None
    arguments = list(segment[1:])
    while arguments and arguments[0] in {"-n", "-e", "-E"}:
        arguments.pop(0)
    template = " ".join(arguments)
    reference = f"${{PIPESTATUS[{pipeline_index}]}}"
    if template.count(reference) != 1:
        return None
    prefix, suffix = template.split(reference, 1)
    if not prefix and not suffix:
        return None
    matches = re.findall(
        rf"(?m)^{re.escape(prefix)}(-?\d+){re.escape(suffix)}\s*$",
        output,
    )
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def explicit_pipeline_test_outcome(
    command: str,
    *,
    observed_output: str,
) -> LogicalCommandOutcome | None:
    """Recover the test producer outcome from a labelled ``PIPESTATUS``.

    This is the selector-free counterpart of :func:`logical_command_outcome`.
    It is needed when lightweight Planning intentionally leaves the concrete
    test address open and the Agent later limits verbose output with a pipe.
    Only the executable test command located by the ordinary command parser is
    eligible; the immediately following shell command must expose the exact
    producer index.  No test-output wording or fuzzy command match is used.
    """

    located = _locate_test_command(command)
    if located is None or not observed_output:
        return None
    segments = _command_segments(located.tokens)
    if not segments:
        return None
    executable_segment = _command_segment_index(
        located.tokens,
        located.executable_index,
    )
    if executable_segment is None:
        return None

    pipeline_start = executable_segment
    while (
        pipeline_start > 0
        and segments[pipeline_start - 1][1] in {"|", "|&"}
    ):
        pipeline_start -= 1
    pipeline_end = executable_segment
    while (
        pipeline_end < len(segments) - 1
        and segments[pipeline_end][1] in {"|", "|&"}
    ):
        pipeline_end += 1
    if pipeline_end == pipeline_start or pipeline_end + 1 >= len(segments):
        return None
    receipt_boundary = segments[pipeline_end][1]
    if receipt_boundary not in {";", "&&", "||"}:
        return None
    pipeline_index = executable_segment - pipeline_start
    exit_code = _echoed_pipeline_exit_code(
        segments[pipeline_end + 1][0],
        observed_output,
        pipeline_index=pipeline_index,
    )
    if exit_code is None:
        return None
    if receipt_boundary == "&&" and exit_code != 0:
        return None
    if receipt_boundary == "||" and exit_code == 0:
        return None
    return LogicalCommandOutcome(
        success=exit_code == 0,
        exit_code=exit_code,
        basis="EXPLICIT_PIPELINE_STATUS_RECEIPT",
    )


def logical_command_outcome(
    selector: str,
    observed_command: str,
    *,
    observed_output: str,
) -> LogicalCommandOutcome | None:
    """Recover a selector-local exit status from an explicit shell receipt.

    A compound Provider command may execute a declared acceptance selector and
    then continue with unrelated work. The aggregate item exit code cannot
    prove the selector's result. When the immediately following shell segment
    emits ``label=$?``, however, POSIX shell semantics provide an independently
    observable result for that exact logical command. This is deterministic
    event normalization, not fuzzy selector matching or a verifier retry.
    """

    selector_tokens = _shell_tokens(_unwrap_shell_script(selector.strip()))
    observed_tokens = _shell_tokens(_unwrap_shell_script(observed_command.strip()))
    if not selector_tokens or not observed_tokens or not observed_output:
        return None
    normalized_selector = tuple(map(str.casefold, selector_tokens))
    segments = _command_segments(observed_tokens)
    for index, (segment, boundary) in enumerate(segments[:-1]):
        if tuple(map(str.casefold, segment)) != normalized_selector:
            continue
        if boundary not in {";", "&&", "||"}:
            continue
        exit_code = _echoed_previous_exit_code(segments[index + 1][0], observed_output)
        if exit_code is None:
            continue
        if boundary == "&&" and exit_code != 0:
            continue
        if boundary == "||" and exit_code == 0:
            continue
        return LogicalCommandOutcome(
            success=exit_code == 0,
            exit_code=exit_code,
            basis="EXPLICIT_SUBCOMMAND_EXIT_RECEIPT",
        )
    return None


def command_selector_matches(
    selector: str,
    observed_command: str,
    *,
    observed_success: bool | None = None,
) -> bool:
    """Resolve one declared command address against a Provider-wrapped command.

    This is deterministic alias translation, not fuzzy retrieval. Shell
    wrappers and quoting are physical transport details; test runner identity,
    target addresses and Python probe ASTs are the semantic command address.
    """

    selector = selector.strip()
    observed_command = observed_command.strip()
    if not selector or not observed_command:
        return False
    selector_script = _unwrap_shell_script(selector)
    observed_script = _unwrap_shell_script(observed_command)
    selector_tokens = _shell_tokens(selector_script)
    observed_tokens = _shell_tokens(observed_script)
    if selector_tokens and tuple(map(str.casefold, selector_tokens)) == tuple(
        map(str.casefold, observed_tokens)
    ):
        return True
    translated_observed = _strip_workspace_cd_transport(observed_tokens)
    if selector_tokens and tuple(map(str.casefold, selector_tokens)) == tuple(
        map(str.casefold, translated_observed)
    ):
        return True
    if observed_success is True and _is_success_implied_and_subchain(
        selector_tokens,
        observed_tokens,
    ):
        return True

    selected = _locate_test_command(selector)
    observed = _locate_test_command(observed_command)
    if selected is None or observed is None:
        return False
    selected_probe = _python_probe_signature(selected)
    observed_probe = _python_probe_signature(observed)
    if selected_probe is not None or observed_probe is not None:
        return selected_probe is not None and selected_probe == observed_probe
    selected_runner, selected_targets = _test_address(selected)
    observed_runner, observed_targets = _test_address(observed)
    if selected_runner == "test-target":
        return bool(selected_targets) and selected_targets.issubset(observed_targets)
    if selected_runner != observed_runner:
        return False
    return not selected_targets or selected_targets.issubset(observed_targets)


def result_selector_matches(
    selector: str,
    *,
    command: str = "",
    command_success: bool | None = None,
    observed_values: tuple[str, ...] = (),
) -> bool:
    """Match a result selector only against typed observation fields."""

    if command_selector_matches(
        selector,
        command,
        observed_success=command_success,
    ):
        return True
    normalized = " ".join(selector.split()).casefold()
    if not normalized:
        return False
    return any(normalized in " ".join(value.split()).casefold() for value in observed_values)


def evidence_result_selector_matches(
    selector: str,
    *,
    canonical_entity_id: str = "",
    content: Mapping[str, object],
) -> bool:
    """Resolve a selector against the typed fields of one Evidence fact."""

    raw_success = content.get("success")
    if isinstance(raw_success, bool):
        command_success: bool | None = raw_success
    elif "exitCode" in content or "exit_code" in content:
        exit_code = content.get("exitCode", content.get("exit_code"))
        command_success = exit_code in (None, 0) and str(
            content.get("status", "completed")
        ) == "completed"
    else:
        command_success = None

    observed_values = tuple(
        str(value)
        for value in (
            canonical_entity_id,
            content.get("testSelector", ""),
            content.get("test_selector", ""),
            content.get("selector", ""),
            content.get("tool_selector", ""),
            content.get("logical_command", ""),
            content.get("tool", ""),
            content.get("name", ""),
            content.get("outcome", ""),
            content.get("aggregatedOutput", ""),
            content.get("output", ""),
            content.get("stdout", ""),
            content.get("stderr", ""),
            content.get("complete_output", ""),
            content.get("output_excerpt", ""),
            content.get("summary", ""),
        )
        if str(value)
    )
    return result_selector_matches(
        selector,
        command=str(content.get("command", "")),
        command_success=command_success,
        observed_values=observed_values,
    )


def command_evidence_semantics(command: str) -> CommandEvidenceSemantics:
    """Classify one observed or planned command using the same local rules."""

    located = _locate_test_command(command)
    if located is None:
        return CommandEvidenceSemantics(False, False)
    trailing_operators = set(located.tokens[located.executable_index + 1 :]).intersection(
        {"|", "|&", "||", ";"}
    )
    reliable = not (
        "||" in trailing_operators
        or ";" in trailing_operators
        or trailing_operators.intersection({"|", "|&"})
        and "pipefail" not in located.script
    )
    return CommandEvidenceSemantics(True, reliable)


_REDIRECT_OPERATOR = re.compile(r"^(?:\d*>>?|\d*<|&>>?)$")
_REDIRECT_WITH_TARGET = re.compile(r"^(?:\d*>>?|\d*<|&>>?)(?:&\d+|&-|\S+)$")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
# Shell preconditions the runtime may repeat verbatim before the test runner.
# Anything else in a ``&&`` prefix (``git stash``, ``pip install``, ``rm``)
# mutates state the model owns, so such a line is never re-observed.
_REPEATABLE_PRECONDITIONS = frozenset(
    {"cd", "export", "source", ".", "set", "unset", "true", "echo", "printf", "pwd", "ls"}
)
# ``git`` subcommands that only read repository state; the model often prints
# the branch or status before running its tests.
_READ_ONLY_GIT_SUBCOMMANDS = frozenset({"status", "log", "rev-parse", "diff", "show", "branch"})
_GIT_BRANCH_MUTATIONS = frozenset({"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy", "-f", "--force", "-u", "--set-upstream-to", "--unset-upstream"})


def _without_redirections(segment: tuple[str, ...]) -> list[str]:
    kept: list[str] = []
    skip_next = False
    for token in segment:
        if skip_next:
            skip_next = False
            continue
        if _REDIRECT_OPERATOR.match(token):
            skip_next = True
            continue
        if _REDIRECT_WITH_TARGET.match(token):
            continue
        kept.append(token)
    return kept


def _repeatable_precondition(segment: tuple[str, ...]) -> bool:
    offset = 0
    while offset < len(segment) and _ENV_ASSIGNMENT.match(segment[offset]):
        offset += 1
    if offset >= len(segment):
        return True
    executable = _basename(segment[offset])
    if executable in _REPEATABLE_PRECONDITIONS:
        return True
    if executable == "git":
        arguments = _without_redirections(segment[offset + 1 :])
        options = [token for token in arguments if token.startswith("-")]
        subcommand = next((token for token in arguments if not token.startswith("-")), "")
        if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
            return False
        if subcommand == "branch" and any(token in _GIT_BRANCH_MUTATIONS for token in options):
            return False
        # ``git branch <name>`` creates a branch; only listing/querying repeats.
        if subcommand == "branch":
            positional = [token for token in arguments if not token.startswith("-")][1:]
            return not positional
        return True
    return False


@dataclass(frozen=True, slots=True)
class CommandObservationScope:
    """What one shell command was addressed at, for directional result binding."""

    test_command: bool
    narrowed: bool
    paths: frozenset[str]


_PATH_LIKE = re.compile(r"^[^\s'\"`]+\.[A-Za-z0-9]{1,6}$")


def command_observation_scope(command: str) -> CommandObservationScope:
    """Describe which paths a command names, ignoring transport wrappers.

    The Provider wraps everything in ``/bin/bash -lc 'cd /app && ...'``; the
    shell binary, ``cd`` targets, wrappers such as ``timeout 300`` and the
    executables themselves are transport, not observation addresses.  In the
    2.2.92 pressure run ``/bin/bash`` and ``/app`` were read as paths, every
    whole-suite ``pytest -n 4`` therefore looked *narrowed* to unrelated
    files and was dropped from the only requirement it verified.

    For a located test command only that command's own arguments decide
    whether it is narrowed (``::`` node ids, ``-k`` expressions, file or
    directory arguments).  For any other command the file-like arguments of
    every segment are the touched paths.
    """

    script = _unwrap_shell_script(command.strip())
    tokens = _shell_tokens(script)
    if not tokens:
        return CommandObservationScope(False, False, frozenset())
    located = _locate_test_command(command)
    chosen: list[tuple[str, ...]] = []
    if located is not None:
        # Only the test runner's own arguments say what it was addressed at.
        chosen.append(located.tokens[located.executable_index + 1 : located.segment_end])
    else:
        for segment, _operator in _command_segments(tokens):
            offset = _skip_command_wrappers(segment, 0)
            if offset >= len(segment):
                continue
            if _basename(segment[offset]) == "cd":
                continue
            chosen.append(segment[offset + 1 :])
    narrowed = False
    paths: set[str] = set()
    for arguments in chosen:
        cleaned_arguments = _without_redirections(arguments)
        for token in cleaned_arguments:
            if token in {"-k", "--keyword"}:
                narrowed = True
                continue
            if token.startswith("-"):
                continue
            candidate = token.strip("'\"`,;()").replace("\\", "/")
            if not candidate or candidate in _COMMAND_BOUNDARIES:
                continue
            if "::" in candidate:
                narrowed = True
                candidate = candidate.split("::", 1)[0]
            if "/" in candidate or _PATH_LIKE.match(candidate):
                paths.add(candidate.removeprefix("./"))
    if located is not None and (paths or narrowed):
        narrowed = True
    if located is None and paths:
        narrowed = True
    return CommandObservationScope(located is not None, narrowed, frozenset(paths))


def reobservable_test_command(command: str) -> str | None:
    """Rebuild the model's own test invocation so its exit status is observable.

    The Provider frequently limits verbose test output with ``| tail -40`` or
    ``| head -80``; the process exit code then belongs to the filter and the
    test outcome is unobservable to the acceptance kernel.  The runtime may
    re-run exactly the located test runner at the current revision: the
    ``&&``-joined preconditions in front of it (``cd``, ``export``, ``source``,
    inline environment assignments) are kept verbatim, every pipe, filter and
    redirection is dropped, and everything after the runner is ignored.  No
    test address is invented -- a line without a recognized test runner, or
    whose preconditions would mutate state, yields ``None``.
    """

    located = _locate_test_command(command)
    if located is None:
        return None
    segments = _command_segments(located.tokens)
    test_index = _command_segment_index(located.tokens, located.executable_index)
    if test_index is None:
        return None
    test_segment = _without_redirections(segments[test_index][0])
    if not test_segment:
        return None
    preconditions: list[tuple[str, ...]] = []
    for index in range(test_index - 1, -1, -1):
        segment, operator = segments[index]
        if operator != "&&":
            break
        if not _repeatable_precondition(segment):
            return None
        preconditions.insert(0, segment)
    parts: list[str] = []
    for segment in preconditions:
        parts.extend(map(shlex.quote, segment))
        parts.append("&&")
    parts.extend(map(shlex.quote, test_segment))
    return " ".join(parts)
