"""Playwright test for the demo login application.

This test is designed to work with Codevid video generation.
Run the Flask app first: python app.py
"""

from playwright.sync_api import Page, expect


def test_successful_login(page: Page) -> None:
    """Test the complete login flow with valid credentials."""
    # Navigate to the login page
    page.goto("http://localhost:5001/login")

    # Wait for the form to be ready
    page.wait_for_selector("#email")

    # Fill in the email field
    page.fill("#email", "demo@example.com")

    # Fill in the password field
    page.fill("#password", "password123")

    # Click the sign in button
    page.click("#submit-btn")

    # Verify we're on the dashboard
    page.wait_for_url("**/dashboard**")

    # Verify the welcome message is visible
    expect(page.locator(".welcome-message")).to_be_visible()
    expect(page.locator(".welcome-message")).to_contain_text("Welcome back, Demo! You have successfully logged in.")
