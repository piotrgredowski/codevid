"""Example Playwright test for a login flow.

This test demonstrates the various Playwright actions that Codevid can parse
and convert into video tutorials.
"""

import re

from playwright.sync_api import Page, expect


def test_login_flow(page: Page) -> None:
    """Test the complete login flow for a user."""
    # Navigate to the login page
    page.goto("https://example.com/login")

    # Wait for the page to be ready
    page.wait_for_load_state("networkidle")

    # Fill in the email field
    page.fill("#email", "user@example.com")

    # Fill in the password field
    page.locator("#password").fill("secretpassword123")

    # Click the login button
    page.click("button[type='submit']")

    # Wait for navigation to complete
    page.wait_for_url("**/dashboard")

    # Verify we're on the dashboard
    expect(page).to_have_title("Dashboard")

    # Verify the welcome message is visible
    expect(page.locator(".welcome-message")).to_be_visible()
    expect(page.locator(".welcome-message")).to_contain_text("Welcome back")


def test_login_with_remember_me(page: Page) -> None:
    """Test login with the remember me checkbox."""
    page.goto("https://example.com/login")

    # Fill credentials
    page.get_by_label("Email").fill("user@example.com")
    page.get_by_label("Password").fill("password123")

    # Check the remember me box
    page.get_by_role("checkbox", name="Remember me").check()

    # Submit the form
    page.get_by_role("button", name="Sign in").click()

    # Verify successful login
    expect(page.get_by_text("Welcome")).to_be_visible()


def test_forgot_password_flow(page: Page) -> None:
    """Test the forgot password functionality."""
    page.goto("https://example.com/login")

    # Click forgot password link
    page.click("text=Forgot password?")

    # Should navigate to reset page
    page.wait_for_url("**/reset-password")

    # Enter email for reset
    page.get_by_placeholder("Enter your email").fill("user@example.com")

    # Click send reset link
    page.click("button:has-text('Send reset link')")

    # Verify confirmation message
    expect(page.locator(".success-message")).to_have_text(
        "Password reset link sent to your email"
    )


async def test_login_async(page: Page) -> None:
    """Test login using async syntax."""
    await page.goto("https://example.com/login")
    await page.fill("#email", "async@example.com")
    await page.fill("#password", "asyncpassword")
    await page.click("#submit")
    await expect(page).to_have_url(re.compile(r".*/dashboard"))
