from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dataset(path: Path):
    from datasets import Dataset, DatasetDict, load_from_disk

    if (path / "dataset_dict.json").is_file() or (path / "state.json").is_file():
        loaded = load_from_disk(str(path))
        return loaded["test"] if isinstance(loaded, DatasetDict) else loaded
    arrow = sorted(path.glob("*/data-*.arrow"))
    if len(arrow) == 1:
        return Dataset.from_file(str(arrow[0]))
    raise FileNotFoundError(f"no single HuggingFace dataset was found under {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--instance-id")
    selection.add_argument("--list-instance-ids", action="store_true")
    args = parser.parse_args()
    dataset = _dataset(args.dataset.resolve())
    if args.list_instance_ids:
        print(
            json.dumps(
                [str(row["instance_id"]) for row in dataset],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 0
    matches = [row for row in dataset if row["instance_id"] == args.instance_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one SWE-EVO instance {args.instance_id!r}, found {len(matches)}"
        )
    # The agent runtime and run artifacts never receive the gold patch.  The
    # official evaluator replaces this field with the model patch later.
    row = dict(matches[0])
    row["patch"] = ""
    print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
