from playwright.sync_api import Page, expect


def test_page_loads(page: Page):
    page.goto("http://localhost:3000", wait_until="commit", timeout=120000)
    expect(page.locator("h1")).to_have_text("Draw.io to RML Converter")
