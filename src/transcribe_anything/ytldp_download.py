"""
Download utility for yt-dlp.
"""

import os
import subprocess

from transcribe_anything.util import PROCESS_TIMEOUT


def ytdlp_download(url: str, outdir: str) -> str:
    """Downloads a file using ytdlp."""
    os.makedirs(outdir, exist_ok=True)
    # remove all files in the directory
    for file in os.listdir(outdir):
        os.remove(os.path.join(outdir, file))

    # Build command as list to avoid shell injection issues
    cmd_list = [
        "yt-dlp",
        "--no-check-certificate",
        url,
        "-o", "out.%(ext)s"
    ]
    cmd_str = subprocess.list2cmdline(cmd_list)
    print(f"Running:\n  {cmd_str}")
    subprocess.run(
        cmd_list,
        shell=False,
        cwd=outdir,
        check=True,
        timeout=PROCESS_TIMEOUT,
        universal_newlines=True,
    )
    new_files = os.listdir(outdir)
    assert len(new_files) == 1, f"Expected 1 file, got {new_files}"
    downloaded_file = os.path.join(outdir, new_files[0])
    assert os.path.exists(downloaded_file), f"The expected file {downloaded_file} doesn't exist"
    return downloaded_file
