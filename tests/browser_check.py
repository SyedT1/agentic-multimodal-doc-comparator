"""Opt-in browser check against the running app; writes local screenshots."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "playwright"))
from playwright.sync_api import sync_playwright, expect


def main():
    folder = ROOT / ".cache" / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1050}, device_scale_factor=1)
        failures = []
        page.on("pageerror", lambda error: failures.append(str(error)))
        page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        expect(page.get_by_role("heading",name="See what changed.")).to_be_visible(timeout=60000)
        expect(page.get_by_role("button",name="Try sample documents",exact=True)).to_be_enabled(timeout=60000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(folder / "desktop-empty.png"), full_page=True)
        page.get_by_role("button",name="Try sample documents",exact=True).click()
        expect(page.get_by_text("Comparison ready. Explore the results below.",exact=True)).to_be_visible(timeout=300000)
        expect(page.get_by_role("heading",name="Comparison overview",exact=True)).to_be_visible()
        expect(page.get_by_role("button",name="Download comparison report",exact=True)).to_be_visible()
        assert page.locator('[data-testid="stException"]').count() == 0
        page.screenshot(path=str(folder / "desktop-results.png"), full_page=True)
        page.get_by_role("tab",name="Documents & changes",exact=True).click()
        expect(page.get_by_role("heading",name="Text changes",exact=True)).to_be_visible()
        expect(page.get_by_role("button",name="Download text diff",exact=True)).to_be_visible()
        page.screenshot(path=str(folder / "desktop-diff.png"), full_page=True)
        page.get_by_role("tab",name="Ask documents",exact=True).click()
        page.get_by_role("button",name="Prepare document chat",exact=True).click()
        expect(page.get_by_placeholder("What changed in the budget or timeline?")).to_be_visible(timeout=180000)
        page.get_by_placeholder("What changed in the budget or timeline?").fill("Compare the budget and delivery schedule in both documents.")
        page.get_by_placeholder("What changed in the budget or timeline?").press("Enter")
        expect(page.locator('[data-testid="stChatMessage"]').last).to_contain_text("18",timeout=180000)
        expect(page.get_by_text("How the answer was found",exact=True)).to_be_visible(timeout=60000)
        assert page.locator('[data-testid="stException"]').count() == 0
        page.screenshot(path=str(folder / "desktop-chat.png"), full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        collapse = page.locator('[data-testid="stSidebarCollapseButton"] button')
        if collapse.count() and collapse.first.is_visible():
            collapse.first.click()
        page.wait_for_timeout(600)
        page.screenshot(path=str(folder / "mobile-chat.png"), full_page=True)
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
        assert not overflow, "The mobile viewport overflows horizontally"
        assert not failures, failures
        print("DESKTOP_MOBILE_COMPARISON_DIFF_LIVE_CHAT_OK", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
