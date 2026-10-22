"""Verify a fresh phone viewport starts with a collapsed sidebar."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH",str(ROOT/".cache"/"playwright"))
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width":390,"height":844},device_scale_factor=1)
    page.goto("http://127.0.0.1:8501",wait_until="networkidle",timeout=60000)
    heading = page.get_by_role("heading",name="See what changed.")
    expect(heading).to_be_visible(timeout=60000)
    expect(page.get_by_role("button",name="Try sample documents",exact=True)).to_be_enabled()
    page.wait_for_timeout(700)
    sidebar = page.locator('[data-testid="stSidebar"]').bounding_box()
    assert sidebar is None or sidebar["x"] + sidebar["width"] <= 2, sidebar
    assert not page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
    assert heading.evaluate("element => element.scrollWidth <= element.clientWidth + 2"), "Heading is clipped"
    assert page.locator('[data-testid="stException"]').count() == 0
    page.screenshot(path=str(ROOT/".cache"/"validation"/"mobile-start.png"),full_page=True)
    print("FRESH_MOBILE_LAYOUT_OK",flush=True)
    browser.close()
