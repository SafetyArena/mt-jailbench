import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # adjust if needed

DIRS_TO_REMOVE = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".eggs",
}

FILES_TO_REMOVE = {".pyc", ".pyo"}


def main():
    removed_dirs = 0
    removed_files = 0

    for path in ROOT.rglob("*"):
        # Remove directories
        if path.is_dir() and path.name in DIRS_TO_REMOVE:
            shutil.rmtree(path, ignore_errors=True)
            removed_dirs += 1

        # Remove files
        elif path.is_file() and path.suffix in FILES_TO_REMOVE:
            try:
                path.unlink()
                removed_files += 1
            except FileNotFoundError:
                pass

    print(f"Removed {removed_dirs} directories and {removed_files} files.")


if __name__ == "__main__":
    main()