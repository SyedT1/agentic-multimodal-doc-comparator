"""Opt-in browser check for actual uploads, report downloads, and batch ranking."""
import csv
import json
import os
import sys
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "playwright"))
from playwright.sync_api import sync_playwright, expect


def main():
    folder = ROOT / ".cache" / "validation"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width":1440,"height":1050})
        page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        expect(page.get_by_role("heading",name="See what changed.")).to_be_visible(timeout=60000)
        if "--batch-only" not in sys.argv:
            uploads = page.locator('input[type="file"]')
            uploads.nth(0).set_input_files(str(folder / "review-1.pdf"))
            uploads.nth(1).set_input_files(str(folder / "review-2.docx"))
            page.get_by_role("button",name="Compare documents",exact=True).click()
            expect(page.get_by_role("heading",name="Comparison overview",exact=True)).to_be_visible(timeout=180000)
            with page.expect_download() as download:
                page.get_by_role("button",name="Download comparison report",exact=True).click()
            report = json.loads(Path(download.value.path()).read_text(encoding="utf-8"))
            assert report["doc1_name"] == "review-1.pdf", report
            assert report["doc2_name"] == "review-2.docx", report
            assert 0 <= report["overall_score"] <= 1
            page.get_by_role("tab",name="Ask documents",exact=True).click()
            page.get_by_role("button",name="Prepare document chat",exact=True).click()
            question = page.get_by_placeholder("What changed in the budget or timeline?")
            expect(question).to_be_visible(timeout=180000)
            question.fill("Compare the budget in both documents.")
            question.press("Enter")
            expect(page.get_by_text("How the answer was found",exact=True)).to_be_visible(timeout=180000)
            answer_block = page.locator('[data-testid="stChatMessage"]').last.locator('[data-testid="stMarkdownContainer"]').filter(has_text=re.compile(r"\S")).first
            expect(answer_block).to_contain_text("12",timeout=180000)
            expect(answer_block).to_contain_text("18",timeout=180000)
            first_answer = answer_block.inner_text()
            assert "12" in first_answer and "18" in first_answer, first_answer
            question.fill("How much did that budget increase?")
            question.press("Enter")
            expect(page.locator('[data-testid="stChatMessage"]')).to_have_count(4,timeout=180000)
            expect(page.get_by_text("How the answer was found",exact=True)).to_be_visible(timeout=180000)
            expect(page.locator('[data-testid="stChatMessage"]').last).to_contain_text("6",timeout=180000)
            answer_block = page.locator('[data-testid="stChatMessage"]').last.locator('[data-testid="stMarkdownContainer"]').filter(has_text=re.compile(r"\S")).first
            expect(answer_block).to_contain_text("6",timeout=180000)
            followup = answer_block.inner_text()
            assert "6" in followup, followup
        page.get_by_text("Batch review",exact=True).click()
        expect(page.get_by_role("heading",name="Find the closest match.")).to_be_visible()
        uploads = page.locator('input[type="file"]')
        uploads.nth(0).set_input_files(str(folder / "review-1.docx"))
        uploads.nth(1).set_input_files([str(folder / "review-1.docx"),str(folder / "review-2.docx")])
        page.get_by_role("button",name="Review collection",exact=True).click()
        expect(page.get_by_role("button",name="Download batch results",exact=True)).to_be_visible(timeout=180000)
        with page.expect_download() as download:
            page.get_by_role("button",name="Download batch results",exact=True).click()
        rows = list(csv.DictReader(Path(download.value.path()).read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2 and all(row["Status"] == "Complete" for row in rows), rows
        assert rows[0]["Document"] == "review-1.docx", rows
        assert float(rows[0]["Similarity"]) > 0.99, rows
        assert page.locator('[data-testid="stException"]').count() == 0
        page.screenshot(path=str(folder / "desktop-batch.png"),full_page=True)
        print("BATCH_RANKING_CSV_OK" if "--batch-only" in sys.argv else
              "PDF_DOCX_UPLOAD_JSON_DOWNLOAD_MULTITURN_CHAT_BATCH_CSV_OK",flush=True)
        browser.close()


if __name__ == "__main__":
    main()
