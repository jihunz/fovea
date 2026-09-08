"""``python -m fovea`` — run the Fovea server."""
from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(prog="fovea", description="Fovea — CV dataset review & labeling workspace")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=None, help="where the index DB and thumbnail cache live")
    ap.add_argument("--model-dir", default=None, help="folder with YOLO .pt weights for auto-label")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    if args.data_dir:
        os.environ["FOVEA_DATA_DIR"] = args.data_dir
    if args.model_dir:
        os.environ["FOVEA_MODEL_DIR"] = args.model_dir
    import uvicorn
    uvicorn.run("fovea.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
