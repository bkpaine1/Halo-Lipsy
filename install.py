"""
Halo-Lipsy Installation Script for ComfyUI Manager
"""

import subprocess
import sys

def install():
    """Install required dependencies"""
    packages = [
        "librosa>=0.9.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "tqdm>=4.60.0",
    ]

    for package in packages:
        print(f"[Halo-Lipsy] Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

    print("[Halo-Lipsy] Installation complete!")
    print("[Halo-Lipsy] Remember to download wav2lip_gan.pth and place in checkpoints/")

if __name__ == "__main__":
    install()
