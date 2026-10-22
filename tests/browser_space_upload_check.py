"""Opt-in upload check inside the actual public Hugging Face iframe."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH",str(ROOT/".cache"/"playwright"))
from playwright.sync_api import sync_playwright, expect

SPACE = "https://huggingface.co/spaces/syedmohaiminulhoque/agentic-doc-sim-streamlit"
HOST = "https://syedmohaiminulhoque-agentic-doc-sim-streamlit.hf.space"


def main():
    artifact = ROOT / ".cache" / "validation"
    artifact.mkdir(parents=True,exist_ok=True)
    sample = artifact / "space-upload.pdf"
    import pymupdf
    with pymupdf.open() as document:
        document.new_page().insert_text((72,72),"Public upload verification sample.")
        document.save(sample)
    responses=[]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width":1440,"height":1050})
        page = context.new_page()
        def capture(response):
            if "/_stcore/upload_file/" in response.url and response.request.method == "PUT":
                responses.append({"status":response.status,"body":response.text()[:200] if response.status >= 400 else "uploaded"})
        page.on("response",capture)
        try:
            page.goto(HOST if "--direct" in sys.argv else SPACE,wait_until="domcontentloaded",timeout=60000)
            target = page if "--direct" in sys.argv else page.frame_locator('iframe[src*=".hf.space"]')
            expect(target.get_by_role("heading",name="See what changed.")).to_be_visible(timeout=120000)
            target.locator('input[type="file"]').nth(0).set_input_files(str(sample))
            page.wait_for_timeout(3000)
            print("UPLOAD_RESPONSES",json.dumps(responses),flush=True)
            cookies=context.cookies(HOST)
            print("XSRF_COOKIE_ATTRIBUTES",json.dumps([{key:c[key] for key in ("name","sameSite","secure")}
                  for c in cookies if c["name"]=="_streamlit_xsrf"]),flush=True)
            assert responses and all(r["status"]==200 or r["status"]==204 for r in responses),responses
            expect(target.get_by_text("space-upload.pdf",exact=True)).to_be_visible(timeout=30000)
            print("HF_IFRAME_UPLOAD_OK" if "--direct" not in sys.argv else "HF_DIRECT_UPLOAD_OK",flush=True)
        except Exception:
            page.screenshot(path=str(artifact/"space-upload-failure.png"),full_page=True)
            print("UPLOAD_RESPONSES",json.dumps(responses),flush=True)
            raise
        finally:
            browser.close()


if __name__=="__main__":
    main()
