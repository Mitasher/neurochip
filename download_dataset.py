"""
Download dataset from Roboflow.

Usage:
    python download_dataset.py --api-key YOUR_ROBOFLOW_API_KEY

The script downloads the 'hat-data-augmentation-ebno9' dataset
in YOLOv8 format and organizes it into helmets.yolov8/ directory.
"""

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Download dataset from Roboflow')
    parser.add_argument(
        '--api-key', type=str, default='djTeG7kblJEVqadsuKN1',
        help='Your Roboflow API key (find it at https://app.roboflow.com/settings/api)'
    )
    parser.add_argument(
        '--workspace', type=str, default='alpos-workspace',
        help='Roboflow workspace name'
    )
    parser.add_argument(
        '--project', type=str, default='hat-data-augmentation-ebno9',
        help='Roboflow project name'
    )
    parser.add_argument(
        '--version', type=int, default=1,
        help='Dataset version number'
    )
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dataset = version.download("yolov8", location="./helmets.yolov8")

    print(f"\nDataset downloaded to: ./helmets.yolov8")
    print("Directory structure:")
    for root, dirs, files in os.walk("./helmets.yolov8"):
        level = root.replace("./helmets.yolov8", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            subindent = " " * 2 * (level + 1)
            for f in files[:5]:
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... ({len(files)} files total)")


if __name__ == "__main__":
    main()
