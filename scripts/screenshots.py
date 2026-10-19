"""Capture the README screenshots with headless Chromium (needs OPENAI_API_KEY).

python scripts/screenshots.py                          # starts its own server on a temp index
python scripts/screenshots.py --url http://localhost:8000   # use a running server
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SAMPLE = ROOT / "sample" / "acme_handbook.pdf"
ANSWERABLE = "How many vacation days do I get, and can I carry unused days over?"
UNANSWERABLE = "What is the CEO's salary?"


@contextmanager
def server(port: int = 8765):
    with tempfile.TemporaryDirectory() as chroma_dir:
        env = {**os.environ, "CHROMA_DIR": chroma_dir}
        cmd = [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(50):
                try:
                    httpx.get(f"{url}/health").raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(0.2)
            yield url
        finally:
            proc.terminate()
            proc.wait()


MAX_HEIGHT = 900


def ask(page: Page, question: str, expect_class: str, output: Path, show_sources: bool) -> None:
    page.fill("#question", question)
    with page.expect_response("**/ask", timeout=60_000):
        page.click("#ask-btn")
    page.wait_for_selector(f"#result{expect_class}:not(.hidden)")
    if show_sources:
        page.click("#sources-title")
    # Crop to the question card + result card, at most MAX_HEIGHT tall.
    top = page.locator("section").nth(1).bounding_box()
    result = page.locator("#result").bounding_box()
    height = min(result["y"] + result["height"] - top["y"] + 16, MAX_HEIGHT)
    page.screenshot(
        path=str(output),
        full_page=True,
        clip={"x": top["x"] - 10, "y": top["y"] - 8, "width": top["width"] + 20, "height": height + 8},
    )
    print(f"Wrote {output.relative_to(ROOT)} ({round(height)}px tall)")


def capture(url: str) -> None:
    with open(SAMPLE, "rb") as f:
        httpx.post(f"{url}/documents", files={"file": (SAMPLE.name, f)}, timeout=120).raise_for_status()
    DOCS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        # CHROMIUM_PATH lets you reuse a system Chromium instead of running `playwright install`.
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH"))
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1.5)
        page.goto(url)
        page.wait_for_selector("#doc option[value]:not([value=''])", state="attached")
        ask(page, ANSWERABLE, ":not(.notfound)", DOCS / "screenshot.png", show_sources=True)
        ask(page, UNANSWERABLE, ".notfound", DOCS / "screenshot-not-found.png", show_sources=False)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="use an already running server instead of starting one")
    args = parser.parse_args()
    if args.url:
        capture(args.url.rstrip("/"))
    else:
        with server() as url:
            capture(url)


if __name__ == "__main__":
    main()
