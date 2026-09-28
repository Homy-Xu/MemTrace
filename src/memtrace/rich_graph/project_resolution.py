"""Resolve only against already indexed files in the SAME graph generation.

This is a syntax-level navigation relation, not runtime dispatch, test success,
or acceptance evidence. No source reads, package downloads, or repository walks.
"""

from __future__ import annotations

import posixpath
from pathlib import Path, PurePosixPath
from typing import Mapping


def rust_module_stem(path: str, module: str, data: Mapping) -> str | None:
    """Resolve Rust modules inside the current or an explicit workspace crate."""
    parts = module.split("::")
    if parts[0] == "crate":
        base = str(data.get("crate_source_root") or "src")
        parts = parts[1:]
    elif parts[0] in {"self", "super"}:
        # foo.rs owns foo/bar.rs; lib.rs/main.rs/mod.rs own siblings.
        p = PurePosixPath(path)
        base = str(p.parent if p.name in {"lib.rs", "main.rs", "mod.rs"} else p.with_suffix(""))
    elif isinstance(data.get("cargo_crate_roots"), Mapping):
        # Cross-crate resolution is allowed only for a crate root read from a
        # workspace member or explicit Cargo path dependency. Registry crates
        # and guessed repository directories remain external.
        crate = parts[0].replace("-", "_")
        roots = data["cargo_crate_roots"]
        root = roots.get(crate)
        if not root:
            return None
        base = str(root)
        parts = parts[1:]
    else:
        return None
    return posixpath.normpath(posixpath.join(base, *(".." if x == "super" else "." if x == "self" else x for x in parts)))


def resolve_project_call(
    path: str, data: Mapping, call: Mapping, files: Mapping[str, Mapping]
) -> tuple[str, str] | None:
    language = data["language"]
    target = str(call["target"])
    if float(call.get("confidence", 1.0)) < 1.0:
        return None
    candidates: set[tuple[str, str]] = set()

    def add(other: str, name: str) -> None:
        index = files.get(other, {})
        other_language = index.get("language")
        compatible = (
            other_language == language
            or {language, other_language} <= {"typescript", "javascript"}
            or {language, other_language} <= {"java", "groovy"}
        )
        if index.get("parser_confidence") != 1 or not compatible:
            return
        if language in {"typescript", "javascript"}:
            first, _, suffix = name.partition(".")
            exported = index.get("exports", {}).get(first)
            if exported:
                name = exported + ("." + suffix if suffix else "")
        names = tuple(str(s["qualified_name"]) for s in index.get("symbols", ()))
        if name in names:
            candidates.add((other, name))
            return
        # JVM overloads have deterministic signature suffixes. Resolve only a
        # unique overload; never pick one by file or reactor order.
        if language in {"java", "groovy"}:
            overloads = tuple(candidate for candidate in names if candidate.startswith(name + "#"))
            if len(overloads) == 1:
                candidates.add((other, overloads[0]))

    def module_paths(stem: str) -> list[str]:
        if stem.startswith("../") or stem.startswith("/"):
            return []
        result = []
        if language in {"typescript", "javascript"}:
            extensions = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
            base = stem[:-3] if stem.endswith(".js") and language == "typescript" else stem
            result = [
                stem,
                *[base + e for e in extensions],
                *[base + "/index" + e for e in extensions],
            ]
        elif language == "rust":
            result = [stem + ".rs", stem + "/mod.rs"]
        return [p for p in result if p in files]

    def alias_stems(value: str) -> list[str]:
        """Resolve only aliases declared by the nearest tsconfig paths map."""

        if language not in {"typescript", "javascript"}:
            return []
        aliases = data.get("path_aliases", {})
        if not isinstance(aliases, Mapping):
            return []
        result: list[str] = []
        for alias, targets in aliases.items():
            alias_text = str(alias)
            prefix = alias_text[:-2] if alias_text.endswith("/*") else alias_text
            if value != prefix and not value.startswith(prefix + "/"):
                continue
            suffix = value[len(prefix) :].lstrip("/")
            values = targets if isinstance(targets, (list, tuple)) else (targets,)
            for target_base in values:
                resolved = str(target_base).replace("*", suffix)
                base_url = str(data.get("tsconfig_base_url") or ".")
                result.append(
                    posixpath.normpath(
                        posixpath.join(base_url, resolved)
                    )
                )
        return result

    if language == "go" and "." not in target:
        for other, index in files.items():
            if PurePosixPath(other).parent == PurePosixPath(path).parent and index.get(
                "module"
            ) == data.get("module"):
                add(other, target)
    if language in {"java", "groovy"}:
        for other, index in files.items():
            if (
                language == "java"
                and index.get("language") == "java"
                and index.get("module") == data.get("module")
            ):
                add(other, target)  # Class.staticMethod, not unknownInstance.method
            if "." in target:
                package_and_class, _, member = target.rpartition(".")
                parts = package_and_class.split(".")
                for split in range(len(parts), 0, -1):
                    package = ".".join(parts[: split - 1])
                    class_name = ".".join(parts[split - 1 :])
                    if index.get("module") == package:
                        add(other, class_name + ("." + member if member else ""))
    if language == "rust" and "::" in target:
        parts = target.split("::")
        name = parts.pop()
        stem = rust_module_stem(path, "::".join(parts), data)
        for other in module_paths(stem) if stem else ():
            add(other, name)
    for imp in data.get("imports", ()):
        alias = imp.get("alias")
        if not alias or alias in {".", "_"} or (alias == "*" and language not in {"java", "groovy"}):
            continue
        if language in {"java", "groovy"} and alias == "*":
            # Static and package wildcards have different meanings.  Never
            # turn a package wildcard into an unqualified method import.
            module = str(imp.get("target", ""))
            if not target:
                continue
            if not imp.get("imported_name"):
                if not module.endswith(".*"):
                    continue
                owner = module[:-2]
                if "." in target:
                    for other, index in files.items():
                        if index.get("module") == owner:
                            add(other, target)
                continue
            if "." in target:
                continue
            owner = module
            owner_package = owner.rsplit(".", 1)[0] if "." in owner else ""
            for other, index in files.items():
                if owner_package and index.get("module") != owner_package:
                    continue
                class_name = owner.rsplit(".", 1)[-1]
                if PurePosixPath(other).stem != class_name:
                    continue
                add(other, class_name + "." + target)
            continue
        rest = ""
        if target == alias:
            rest = str(imp.get("imported_name") or ("default" if language in {"typescript", "javascript"} else ""))
        elif target.startswith((alias + ".", alias + "::")):
            rest = target[len(alias) :].lstrip(".:").replace("::", ".")
            if imp.get("imported_name"):
                rest = str(imp["imported_name"]) + "." + rest
        else:
            continue
        module = str(imp["target"])
        if language in {"typescript", "javascript"} and module.startswith("."):
            stem = posixpath.normpath(posixpath.join(str(PurePosixPath(path).parent), module))
            for other in module_paths(stem):
                add(other, rest)
        elif language in {"typescript", "javascript"}:
            for stem in alias_stems(module):
                for other in module_paths(stem):
                    add(other, rest)
        elif language == "go":
            prefix = data.get("module_prefix")
            if prefix and (module == prefix or module.startswith(prefix + "/")):
                directory = module[len(prefix) :].strip("/") or "."
                for other in files:
                    if str(PurePosixPath(other).parent) == directory:
                        add(other, rest)
        elif language in {"java", "groovy"}:
            for other, index in files.items():
                if index.get("language") not in {"java", "groovy"}:
                    continue
                package = str(index.get("module", ""))
                class_name = PurePosixPath(other).stem
                full_class = package + "." + class_name if package else class_name
                if module == full_class:
                    # A type/annotation reference may name the imported class
                    # itself (``@Marker`` or a Groovy coverage hint), while a
                    # constructor/method reference carries a member suffix.
                    add(other, class_name + ("." + rest if rest else ""))
                elif module.startswith(full_class + "."):
                    add(
                        other,
                        module[len(full_class) + 1 :] if "." in rest else class_name + "." + rest,
                    )
        elif language == "rust":
            stem = rust_module_stem(path, module, data)
            for other in module_paths(stem) if stem else ():
                add(other, rest)
    if language in {"typescript", "javascript"}:
        for stem in alias_stems(target):
            for other in module_paths(stem):
                add(other, target.rsplit(".", 1)[-1])
    candidates.discard((path, target))
    return next(iter(candidates)) if len(candidates) == 1 else None


def dependency_paths(root: Path, path: str, data: Mapping, *, limit: int = 16) -> tuple[str, ...]:
    """Bounded one-hop imports from a frontier file, never a repository walk.

    Mirrors Python's repository-local module binding without guessing external
    packages or type dispatch. The worker may index these neighbours once.
    """
    language = data.get("language")
    if data.get("parser_confidence") != 1 and language != "groovy":
        return ()
    parent = PurePosixPath(path).parent
    stems: list[str] = []
    candidates: list[str] = []
    for imp in data.get("imports", ())[:limit]:
        module = str(imp["target"])
        if language in {"typescript", "javascript"} and module.startswith("."):
            stems.append(posixpath.normpath(str(parent / module)))
        elif language in {"typescript", "javascript"}:
            aliases = data.get("path_aliases", {})
            if isinstance(aliases, Mapping):
                for alias, targets in aliases.items():
                    alias_text = str(alias)
                    prefix = (
                        alias_text[:-2]
                        if alias_text.endswith("/*")
                        else alias_text
                    )
                    if module != prefix and not module.startswith(prefix + "/"):
                        continue
                    suffix = module[len(prefix) :].lstrip("/")
                    values = (
                        targets
                        if isinstance(targets, (list, tuple))
                        else (targets,)
                    )
                    for target_base in values:
                        resolved = str(target_base).replace("*", suffix)
                        base_url = str(data.get("tsconfig_base_url") or ".")
                        stems.append(
                            posixpath.normpath(
                                posixpath.join(base_url, resolved)
                            )
                        )
        elif language == "rust":
            stem = rust_module_stem(path, module, data)
            if stem:
                stems.append(stem)
        elif language in {"java", "groovy"}:
            package = str(data.get("module", ""))
            # Static imports name a member (or '*') after the declaring
            # class.  The frontier neighbour is that class's source file.
            base = parent
            for _ in package.split(".") if package else ():
                base = base.parent
            if module.endswith(".*"):
                imported_package = module[:-2]
                directory = parent if imported_package == package else base / imported_package.replace(".", "/")
                folder = (root / directory).resolve()
                if folder.is_relative_to(root.resolve()):
                    try:
                        from itertools import islice
                        candidates.extend(
                            child.relative_to(root.resolve()).as_posix()
                            for child in islice(folder.iterdir(), 256) if child.suffix == ".java"
                        )
                    except OSError:
                        pass
                continue
            # javac also permits sibling source files outside a conventional
            # package-directory layout; binding still checks package + class.
            if package and module.startswith(package + "."):
                candidates.append(str(parent / (module[len(package) + 1:] + ".java")))
            candidates.append(str(base / (module.replace(".", "/") + ".java")))
            # Admit generated Java only when a declared Maven generator owns
            # the source root and the offline build has actually created it.
            try:
                from memtrace.swe_milestone.maven_reactor import MavenReactor

                reactor = MavenReactor.load(root)
                owner = reactor.nearest_module(path)
                if owner is not None and not module.endswith(".*"):
                    relative_class = module.replace(".", "/") + ".java"
                    # Generated types commonly live in an upstream Maven
                    # module. Traverse only the declared reactor dependency
                    # closure; never scan arbitrary target/ directories.
                    for reactor_module in reactor.affected_modules(
                        (path,), upstream_depth=8, downstream_depth=0, max_modules=32
                    ):
                        for generated_root in reactor.existing_generated_roots(
                            reactor_module.path
                        ):
                            candidate = generated_root / relative_class
                            try:
                                candidates.append(candidate.relative_to(root).as_posix())
                            except ValueError:
                                continue
            except (OSError, ValueError):
                pass
    if language == "go":
        directories = {str(parent)}
        prefix = data.get("module_prefix")
        if prefix:
            for imp in data.get("imports", ())[:limit]:
                module = str(imp["target"])
                if module.startswith(prefix + "/"):
                    directories.add(module[len(prefix) + 1:])
        for directory in sorted(directories)[:limit]:
            folder = (root / directory).resolve()
            if not folder.is_relative_to(root.resolve()):
                continue
            try:
                # One package directory, bounded entries; no recursive scan.
                from itertools import islice
                for child in islice(folder.iterdir(), 256):
                    if child.suffix == ".go":
                        candidates.append(child.relative_to(root).as_posix())
            except OSError:
                continue
    for stem in stems:
        if language in {"typescript", "javascript"}:
            base = stem[:-3] if stem.endswith(".js") else stem
            candidates.extend([stem, *(base + ext for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")),
                               *(base + "/index" + ext for ext in (".ts", ".tsx", ".js"))])
        else:
            candidates.extend((stem + ".rs", stem + "/mod.rs"))
    result = []
    for candidate in dict.fromkeys(candidates):
        absolute = (root / candidate).resolve()
        if absolute.is_relative_to(root.resolve()) and absolute.is_file():
            relative = absolute.relative_to(root.resolve()).as_posix()
            if relative != path and relative not in result:
                result.append(relative)
            if len(result) == limit:
                break
    return tuple(result)
