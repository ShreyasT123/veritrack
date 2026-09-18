import os
import sys

EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".sql",".yaml",".yml",".json",".txt"}

def count_lines(directory):
    counts = {ext: 0 for ext in EXTENSIONS}
    files = {ext: 0 for ext in EXTENSIONS}

    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()

            if ext not in EXTENSIONS:
                continue

            path = os.path.join(root, filename)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = sum(1 for _ in f)

                counts[ext] += lines
                files[ext] += 1

            except (OSError, UnicodeError):
                pass

    print(f"{'Extension':<10} {'Files':>8} {'Lines':>12}")
    print("-" * 32)

    total_files = 0
    total_lines = 0

    for ext in sorted(EXTENSIONS):
        print(f"{ext:<10} {files[ext]:>8} {counts[ext]:>12}")
        total_files += files[ext]
        total_lines += counts[ext]

    print("-" * 32)
    print(f"{'TOTAL':<10} {total_files:>8} {total_lines:>12}")


directory = sys.argv[1] if len(sys.argv) > 1 else "."

count_lines(directory)