from playwright.sync_api import Page, expect

def test_google_search(page: Page):
    # Navigate to Google
    page.goto("https://www.google.pl")
    
    # Accept cookies if the dialog appears (optional, based on region)
    # Using a broad selector that might match "Reject all" or "Accept all"
    # This is a common pattern for Google consent popups
    # try:
    #     page.get_by_role("button", name="Odrzuć wszystko").click(timeout=2000)
    # except:
    #     pass

    # Search for "Codevid"
    search_input = page.locator("textarea[name='q']")
    search_input.fill("Codevid python")
    search_input.press("Enter")

    # Wait for results
    page.wait_for_selector("#search")
    
    # Verify results contain expected text
    expect(page.locator("#search")).to_contain_text("Codevid")
