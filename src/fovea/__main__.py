"""``python -m fovea`` — run the Fovea server."""
from __future__ import annotations

import argparse
import ipaddress
import os
import sys


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(prog="fovea", description="Fovea — CV dataset review & labeling workspace")
    # Loopback by default: the API has no authentication and can browse the whole filesystem.
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other machines (no auth — trusted networks only; "
                                                         "set FOVEA_TOKEN to require an access token from them)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=None, help="where the index DB and thumbnail cache live")
    ap.add_argument("--model-dir", default=None, help="folder with model weights for AI labeling")
    ap.add_argument("--allowed-hosts", default=None, help="extra host names the server answers to, comma-separated "
                                                          "(e.g. fovea.lab.example); IP addresses always work")
    ap.add_argument("--allow-remote-ai", action="store_true",
                    help="let other machines use cloud AI providers with the keys stored here (spends your money)")
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    if args.data_dir:
        os.environ["FOVEA_DATA_DIR"] = args.data_dir
    if args.model_dir:
        os.environ["FOVEA_MODEL_DIR"] = args.model_dir
    if args.allowed_hosts:
        os.environ["FOVEA_ALLOWED_HOSTS"] = ",".join(filter(None, [os.environ.get("FOVEA_ALLOWED_HOSTS", ""), args.allowed_hosts]))
    if args.allow_remote_ai:
        os.environ["FOVEA_ALLOW_REMOTE_AI"] = "1"
    os.environ["FOVEA_BIND_HOST"] = args.host
    if not _is_loopback(args.host) and not os.environ.get("FOVEA_TOKEN"):
        if os.path.exists("/.dockerenv"):
            print("Fovea in Docker: who can connect is decided by the published port (docker-compose.yml binds "
                  "127.0.0.1). Set FOVEA_TOKEN before publishing it to your network.", file=sys.stderr)
        else:
            print(f"Fovea is listening on {args.host}: anyone who can reach this port can browse files under it and "
                  "edit labels. Set FOVEA_TOKEN=<secret> to require an access token from other machines.", file=sys.stderr)

    import uvicorn

    from .jobs import jobs

    class _Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            # Before uvicorn starts waiting for open connections: end job event streams and cancel jobs,
            # so an open browser tab or a long export cannot hold the shutdown.
            jobs.begin_shutdown()
            super().handle_exit(sig, frame)

    config = uvicorn.Config("fovea.main:app", host=args.host, port=args.port, reload=args.reload,
                            timeout_graceful_shutdown=5)
    if args.reload:
        uvicorn.run("fovea.main:app", host=args.host, port=args.port, reload=True, timeout_graceful_shutdown=5)
        return
    try:
        _Server(config).run()
    except KeyboardInterrupt:
        # uvicorn re-raises a captured Ctrl+C after its clean shutdown; uvicorn.run() swallows it, and so
        # must we, or a normal stop ends with a KeyboardInterrupt traceback.
        pass


if __name__ == "__main__":
    main()
