"""
Standalone script — sirf cache banana ho to yeh chalaao.
train.py already internally call karta hai make_motion_cache().
Yeh script tab use karo jab sirf preprocessing test karni ho.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.train import make_motion_cache, class_names

VIDEO_ROOT = os.environ.get(
    "VIDEO_ROOT",
    "/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset/Real Life Violence Dataset"
)
CACHE_DIR = os.environ.get("CACHE_DIR", "/kaggle/working/violence_motion_cache_for_cl")

if __name__ == "__main__":
    print(f"Building motion cache: {CACHE_DIR}")
    make_motion_cache(VIDEO_ROOT, CACHE_DIR, classes=class_names,
                      frames_per_video=16, debug=True)
    print("Cache build complete.")