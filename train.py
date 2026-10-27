"""Fine-tune the final checkpoint with the official Ultralytics trainer."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset YAML; 0=smoke, 1=fire")
    parser.add_argument("--model", default=str(Path(__file__).parent / "models" / "best.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    YOLO(args.model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                           batch=args.batch, device=args.device, workers=0,
                           project="runs/train", name="fire-smoke")

if __name__ == "__main__":
    main()
