#!/usr/bin/env python3
import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Set, Tuple
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a zip archive of a workspace with flexible exclusions. "
            "By default, excludes .vscode, runs_l40, cache/.cache, __pycache__, and all .csv files."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path.cwd()),
        help="Root directory to archive (default: current working directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="lab3.zip",
        help="Output zip file name (default: lab3.zip)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Glob pattern(s) to exclude (match against POSIX-style relative paths). "
            "Can be provided multiple times. Example: --exclude 'runs/*' --exclude '**/*.tmp'"
        ),
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help=(
            "Directory name(s) or glob(s) to exclude anywhere in the tree. "
            "Match is against the directory name and its relative path. Can be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-ext",
        action="append",
        default=[],
        help=(
            "File extension(s) to exclude (e.g., .csv). Can be repeated. Case-insensitive."
        ),
    )
    parser.add_argument(
        "--include-git-tracked-only",
        action="store_true",
        help="Only include files tracked by git (ignores untracked).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not create the archive; only print what would be included/excluded.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose details about included files.",
    )

    return parser.parse_args()


def normalize_patterns(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in patterns:
        if not p:
            continue
        out.append(p.replace("\\", "/"))
    return out


def get_git_tracked_files(root: Path) -> Set[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        files = {root / Path(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        return files
    except Exception:
        return set()


def should_exclude_dir(rel_dir_posix: str, dir_name: str, exclude_dir_patterns: List[str]) -> bool:
    for pat in exclude_dir_patterns:
        if fnmatch.fnmatch(dir_name, pat) or fnmatch.fnmatch(rel_dir_posix, pat):
            return True
    return False


def should_exclude_file(rel_path_posix: str, ext: str, exclude_exts: Set[str], exclude_path_patterns: List[str]) -> bool:
    if ext.lower() in exclude_exts:
        return True
    for pat in exclude_path_patterns:
        if fnmatch.fnmatch(rel_path_posix, pat):
            return True
    return False


def walk_files(
    root: Path,
    exclude_dirs: List[str],
    exclude_path_patterns: List[str],
    exclude_exts: Set[str],
    git_tracked_only: bool,
) -> Tuple[List[Path], List[Path]]:
    included: List[Path] = []
    excluded: List[Path] = []

    tracked: Set[Path] = set()
    if git_tracked_only:
        tracked = get_git_tracked_files(root)

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_p = Path(dirpath)
        rel_dir = dirpath_p.relative_to(root)
        rel_dir_posix = rel_dir.as_posix() if str(rel_dir) != "." else ""

        pruned = []
        for i in reversed(range(len(dirnames))):
            d = dirnames[i]
            dir_rel_path_posix = (rel_dir / d).as_posix() if rel_dir_posix else d
            if should_exclude_dir(dir_rel_path_posix, d, exclude_dirs):
                pruned.append(dirnames[i])
                del dirnames[i]
        # files
        for f in filenames:
            full = dirpath_p / f
            if git_tracked_only and full not in tracked:
                excluded.append(full)
                continue
            rel_path = full.relative_to(root)
            rel_path_posix = rel_path.as_posix()
            if rel_path_posix.startswith("./"):
                rel_path_posix = rel_path_posix[2:]
            if should_exclude_file(rel_path_posix, full.suffix, exclude_exts, exclude_path_patterns):
                excluded.append(full)
                continue
            included.append(full)
    return included, excluded


def create_zip(root: Path, output: Path, files: List[Path]) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.relative_to(root).as_posix())


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    default_exclude_dirs = [
        ".vscode",
        "runs_l40",
        "__pycache__",
        ".cache",
        "cache",
    ]
    default_exclude_exts = [".csv"]

    exclude_dir_patterns = normalize_patterns(default_exclude_dirs + (args.exclude_dir or []))
    exclude_path_patterns = normalize_patterns(args.exclude or [])
    exclude_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (default_exclude_exts + (args.exclude_ext or []))}

    included, excluded = walk_files(
        root=root,
        exclude_dirs=exclude_dir_patterns,
        exclude_path_patterns=exclude_path_patterns,
        exclude_exts=exclude_exts,
        git_tracked_only=args.include_git_tracked_only,
    )

    if args.verbose or args.dry_run:
        print(f"Root: {root}")
        print(f"Output: {output}")
        print(f"Exclude dirs: {exclude_dir_patterns}")
        print(f"Exclude patterns: {exclude_path_patterns}")
        print(f"Exclude extensions: {sorted(exclude_exts)}")
        print(f"Git-tracked only: {args.include_git_tracked_only}")
        print(f"Included files: {len(included)} | Excluded files: {len(excluded)}")
        if args.verbose:
            for p in included[:50]:
                print(f"+ {p.relative_to(root).as_posix()}")
            if len(included) > 50:
                print(f"... ({len(included) - 50} more)")
    if args.dry_run:
        return 0

    if not included:
        print("No files to archive after applying filters.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    create_zip(root, output, included)
    print(f"Archive created: {output}")
    print(f"Files included: {len(included)} | Files excluded: {len(excluded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
