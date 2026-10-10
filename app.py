"""
🌾 UZHAVAN AI - Unified Application Launcher
Simple, single-command runner for UZHAVAN AI.

Usage:
    python app.py                 # Run full system (Web UI + Backend API on http://localhost:8000)
    python app.py --dev           # Run in dev mode (FastAPI backend + Vite live-reload frontend)
    python app.py --port 8080     # Custom port
    python app.py --no-browser    # Do not auto-open web browser
"""

import os
import sys
import time
import socket
import argparse
import subprocess
import webbrowser
import threading
from pathlib import Path

# Reconfigure stdout/stderr for Windows UTF-8 console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Auto-detect project virtualenv and re-exec if running with an incompatible global Python (e.g. Python 3.14)
VENV_PYTHON = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if VENV_PYTHON.exists():
    try:
        current_exe = Path(sys.executable).resolve()
        target_exe = VENV_PYTHON.resolve()
        if current_exe != target_exe and not os.environ.get("_UZHAVAN_VENV_ACTIVE"):
            os.environ["_UZHAVAN_VENV_ACTIVE"] = "1"
            res = subprocess.run([str(target_exe), *sys.argv])
            sys.exit(res.returncode)
    except Exception:
        pass


def find_available_port(host: str, preferred_port: int, max_attempts: int = 15) -> int:
    """Find an available port starting from preferred_port."""
    for p in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return preferred_port


def print_banner(host: str, port: int, dev_mode: bool = False):
    banner = f"""
======================================================================
🌾 UZHAVAN AI — Rural Tamil Dialect Voice Assistant
   Dialect-Aware AI for Farmers & Elderly Citizens (Research Edition)
======================================================================
  🌐 Web Application   : http://localhost:{port}
  📚 API Swagger Docs  : http://localhost:{port}/docs
  ❤️ Health Check      : http://localhost:{port}/api/health
  🔬 Research Benchmark: http://localhost:{port}/api/research/benchmark-summary
"""
    if dev_mode:
        banner += f"  ⚡ Vite Dev Server   : http://localhost:5173\n"
    banner += """----------------------------------------------------------------------
  Press Ctrl + C to stop the application.
======================================================================
"""
    print(banner)


def open_browser_delayed(url: str, delay: float = 1.5):
    """Wait for server to start, then open the browser."""
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


def check_frontend_dist():
    """Ensure frontend static files exist; build if necessary."""
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if not (dist_dir / "index.html").exists():
        print("⚙️  Frontend build not found. Compiling React frontend (npm run build)...")
        try:
            npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
            subprocess.run(
                [npm_cmd, "run", "build"],
                cwd=str(PROJECT_ROOT / "frontend"),
                check=True,
                capture_output=True,
                text=True,
            )
            print("✅ Frontend compiled successfully!")
        except Exception as e:
            print(f"⚠️  Could not compile frontend automatically: {e}")
            print("   You can still run backend API and use frontend in dev mode (`python app.py --dev`)")


def run_dev_mode(host: str, port: int, open_browser: bool = True):
    """Run backend and frontend Vite server concurrently."""
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"

    print("🚀 Starting Vite React Frontend on http://localhost:5173...")
    vite_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(PROJECT_ROOT / "frontend"),
    )

    if open_browser:
        open_browser_delayed("http://localhost:5173", delay=2.0)

    print(f"🚀 Starting FastAPI Backend on http://{host}:{port}...")
    import uvicorn
    try:
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 Stopping UZHAVAN AI...")
    finally:
        vite_proc.terminate()
        try:
            vite_proc.wait(timeout=2.0)
        except Exception:
            pass


def run_production_mode(host: str, port: int, open_browser: bool = True):
    """Run unified FastAPI application serving both React SPA and Backend API."""
    check_frontend_dist()

    url = f"http://localhost:{port}"
    if open_browser:
        open_browser_delayed(url, delay=1.2)

    import uvicorn
    try:
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n👋 Stopping UZHAVAN AI...")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="🌾 UZHAVAN AI Application Runner")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Server port (default: 8000)")
    parser.add_argument("--dev", action="store_true", help="Launch Vite dev server concurrently (live hot-reload)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    # Check port availability
    target_port = find_available_port(args.host, args.port)
    if target_port != args.port:
        print(f"ℹ️  Port {args.port} is already in use. Automatically switched to port {target_port}!")
    
    print_banner(args.host, target_port, dev_mode=args.dev)

    if args.dev:
        run_dev_mode(args.host, target_port, open_browser=not args.no_browser)
    else:
        run_production_mode(args.host, target_port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
