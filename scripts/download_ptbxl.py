#!/usr/bin/env python3
"""
PTB-XL Dataset Download Script for MOSAIC-Med
Downloads the official PTB-XL dataset (version 1.0.3) from PhysioNet.

Reference:
Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset",
Scientific Data, 2020. https://physionet.org/content/ptb-xl/1.0.3/
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil

PHYSIONET_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
TARGET_DIR = "ptbxl"

def download_via_wfdb(target_dir):
    try:
        import wfdb
        print(f"Downloading PTB-XL (v1.0.3) via wfdb into '{target_dir}'...")
        wfdb.dl_database('ptb-xl', target_dir)
        print("Download via wfdb completed successfully.")
        return True
    except Exception as e:
        print(f"wfdb download encountered an error: {e}")
        return False

def download_via_http(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "ptbxl.zip")
    print(f"Downloading PTB-XL archive from {PHYSIONET_URL}...")
    
    def reporthook(blocknum, blocksize, totalsize):
        read = blocknum * blocksize
        if totalsize > 0:
            percent = read * 100 / totalsize
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({read // (1024*1024)} MB / {totalsize // (1024*1024)} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(PHYSIONET_URL, zip_path, reporthook=reporthook)
        print("\nDownload finished. Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(zip_path)
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"\nDirect HTTP download encountered an error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download PTB-XL v1.0.3 from PhysioNet")
    parser.add_argument("--target_dir", default=TARGET_DIR, help="Destination directory (default: ptbxl)")
    parser.add_argument("--method", choices=["wfdb", "http", "auto"], default="auto", help="Download method")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.target_dir, "ptbxl_database.csv")):
        print(f"PTB-XL is already present at '{args.target_dir}'.")
        return

    success = False
    if args.method in ["wfdb", "auto"]:
        success = download_via_wfdb(args.target_dir)
    if not success and args.method in ["http", "auto"]:
        print("Falling back to direct HTTP archive download...")
        success = download_via_http(args.target_dir)

    if success:
        print(f"\n[OK] PTB-XL dataset ready in '{args.target_dir}'.")
    else:
        print("\n[ERROR] Automated download failed. Please manually download PTB-XL v1.0.3 from:")
        print("https://physionet.org/content/ptb-xl/1.0.3/ and extract into 'ptbxl/'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
