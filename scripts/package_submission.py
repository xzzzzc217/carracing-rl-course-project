from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


INCLUDE_PATHS = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "pyproject.toml",
    "configs",
    "src",
    "scripts",
    "tests",
    "report",
    "results",
]


def should_include(path: Path) -> bool:
    excluded_suffixes = {".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}
    excluded_parts = {"__pycache__", ".pytest_cache", "tmp", "smoke_seed42"}
    if path.name.startswith("."):
        return False
    if any(str(path).endswith(suffix) for suffix in excluded_suffixes):
        return False
    if path.suffix == ".pt":
        return False
    if excluded_parts.intersection(path.parts):
        return False
    if any(part.endswith(".egg-info") for part in path.parts):
        return False
    return True


def add_path(zip_file: zipfile.ZipFile, path: Path, root: Path) -> None:
    if path.is_file() and should_include(path):
        relative_path = path.relative_to(root)
        if relative_path == Path("report/main.pdf") and (root / "report/main_updated.pdf").exists():
            return
        if relative_path == Path("report/main_updated.pdf"):
            # Some Windows PDF viewers lock main.pdf; package the fresh build under the expected name.
            zip_file.write(path, arcname=Path("report/main.pdf"))
            return
        zip_file.write(path, arcname=relative_path)
    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file() and should_include(child):
                relative_path = child.relative_to(root)
                if relative_path == Path("report/main.pdf") and (root / "report/main_updated.pdf").exists():
                    continue
                if relative_path == Path("report/main_updated.pdf"):
                    zip_file.write(child, arcname=Path("report/main.pdf"))
                    continue
                zip_file.write(child, arcname=relative_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="57123117赵子辰", help="Zip filename stem required by the course.")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    root = Path.cwd()
    output_path = Path(args.output_dir) / f"{args.student}.zip"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for relative in INCLUDE_PATHS:
            path = root / relative
            if path.exists():
                add_path(zip_file, path, root)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
