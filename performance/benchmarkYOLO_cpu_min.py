import argparse
import time
import numpy as np
from ultralytics import YOLO
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Minimal YOLO CPU benchmark")
    parser.add_argument("weights", help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=160, help="Inference image size")
    parser.add_argument("--seconds", type=float, default=10.0, help="Seconds to run")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Minimal synthetic input to avoid camera/file I/O overhead
    img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)

    duration_s = args.seconds

    print(f"CPU benchmark running for {duration_s:.2f}s...")
    start = time.perf_counter()
    with open("CPU.txt", "w", encoding="utf-8") as f:
        while True:
            model.predict(img, imgsz=args.imgsz, device="cpu", verbose=False)
            elapsed = time.perf_counter() - start
            if elapsed >= duration_s:
                break
            f.write(f"{elapsed:.6f}\n")
    print("CPU benchmark complete.")

    print(f"GPU benchmark running for {duration_s:.2f}s...")
    start = time.perf_counter()
    with open("GPU.txt", "w", encoding="utf-8") as f:
        while True:
            model.predict(img, imgsz=args.imgsz, verbose=False, device=DEVICE)
            elapsed = time.perf_counter() - start
            if elapsed >= duration_s:
                break
            f.write(f"{elapsed:.6f}\n")
    print("GPU benchmark complete.")


if __name__ == "__main__":
    main()
