#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package Bundler Script
Downloads specified packages as wheels to a local directory.
Simulates "downloading from elsewhere" for offline installation.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('bundler')

def download_packages(packages, output_dir, pip_args=None):
    logger = logging.getLogger('bundler')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading packages to: {output_path}")
    logger.info(f"Packages: {', '.join(packages)}")
    
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(output_path)
    ]
    
    if pip_args:
        cmd.extend(pip_args)
        
    cmd.extend(packages)
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Download completed successfully.")
        
        # List downloaded files
        files = list(output_path.glob("*"))
        logger.info(f"Downloaded {len(files)} files:")
        for f in files:
            logger.info(f"  - {f.name}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download Python packages as wheels.")
    parser.add_argument("--packages", nargs="+", required=True, help="List of packages to download")
    parser.add_argument("--output", "-o", default="packages", help="Output directory (default: packages)")
    parser.add_argument("--platform", help="Platform tag (e.g., manylinux1_x86_64) for cross-platform download")
    parser.add_argument("--python-version", help="Python version (e.g., 38 for 3.8)")
    
    args = parser.parse_args()
    
    pip_args = []
    if args.platform:
        pip_args.extend(["--platform", args.platform, "--only-binary=:all:"])
    if args.python_version:
        pip_args.extend(["--python-version", args.python_version])
        
    setup_logging()
    download_packages(args.packages, args.output, pip_args)

if __name__ == "__main__":
    main()
