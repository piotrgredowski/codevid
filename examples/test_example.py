from playwright.sync_api import Page, expect

def test_example_domain(page: Page):
    """Test navigation to example.org"""
    # Navigate to the example domain
    page.goto("https://example.org/")
    
    # Verify the title
    expect(page).to_have_title("Example Domain")
    
    # Verify the main header
    expect(page.locator("h1")).to_contain_text("Example Domain")
    
    # Verify the paragraph text
    expect(page.locator("p").first).to_contain_text("This domain is for use in illustrative examples")
    
    # Click the "More information" link
    page.get_by_role("link", name="More information").click()
    
    # Verify we navigated to IANA
    expect(page).to_have_url("https://www.iana.org/help/example-domains")
