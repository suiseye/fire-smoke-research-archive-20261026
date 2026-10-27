"""Run the final fire/smoke checkpoint with the official Ultralytics API."""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Image, video, directory or camera index")
    parser.add_argument("--model", default=str(Path(__file__).parent / "models" / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdecimal() else args.source
    results = YOLO(args.model).predict(source=source, imgsz=args.imgsz, conf=args.conf,
                                      device=args.device, stream=True, save=True,
                                      project="runs/predict", name="fire-smoke")
    for _ in results:
        pass

if __name__ == "__main__":
    main()
