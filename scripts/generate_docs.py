import subprocess


def main():
    # Run pdoc with predefined arguments
    subprocess.run(
        [
            "pdoc",
            "-o",
            "docs",
            "-d",
            "google",
            "--no-include-undocumented",
            "magic_eraser",
        ],
        check=True,
    )
