"""
One-click launcher for the ResVal House Price dashboard.

    python run_dashboard.py            -> opens the dashboard (runs the ML pipeline first if needed)
    python run_dashboard.py --refresh  -> re-trains the models, then opens the dashboard

No localhost, XAMPP or web server needed: the dashboard is a plain HTML file that the
browser opens straight from disk.

Keep these files in the SAME folder as this script:
    house_price.py   (or python_updated.txt)   - the ML pipeline
    dashboard.html   (or html_updated.txt)     - the dashboard page
"""
import os
import sys
import shutil
import subprocess
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE_NAMES = ["house_price.py", "python_updated.py", "python_updated.txt"]
HTML_NAMES = ["dashboard.html", "html_updated.html", "html_updated.txt"]
DATA_FILE = HERE / "dashboard_data.js"


def find(names, label):
    for name in names:
        p = HERE / name
        if p.exists():
            return p
    sys.exit(f"[x] Could not find your {label}. Put one of these next to run_dashboard.py: {', '.join(names)}")


def main():
    html = find(HTML_NAMES, "dashboard HTML file")
    if html.suffix.lower() != ".html":              # e.g. html_updated.txt -> dashboard.html
        target = HERE / "dashboard.html"
        shutil.copyfile(html, target)
        html = target

    if "--refresh" in sys.argv or not DATA_FILE.exists():
        pipeline = find(PIPELINE_NAMES, "Python pipeline file")
        print(f"[1/2] Training models with {pipeline.name} (this takes a minute)...")
        env = dict(os.environ, OUT_DIR=str(HERE), MPLBACKEND="Agg")
        result = subprocess.run([sys.executable, str(pipeline)], cwd=HERE, env=env)
        if result.returncode != 0 or not DATA_FILE.exists():
            sys.exit("[x] The pipeline failed. Install the libraries with:\n"
                     "    pip install numpy pandas matplotlib seaborn scikit-learn")
    else:
        print("[1/2] Using existing results (run with --refresh to re-train).")

    url = html.resolve().as_uri()
    print(f"[2/2] Opening dashboard: {url}")
    if not webbrowser.open(url):
        print("    Could not launch a browser automatically - open that file manually.")


if __name__ == "__main__":
    main()