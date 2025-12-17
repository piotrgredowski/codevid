


"""Playwright test for openmercato application.
"""

from playwright.sync_api import Page, expect


def test_open_mercato_customer_creation(page: Page) -> None:
    page.goto("http://localhost:3000/login?role=admin")
    page.wait_for_selector("#email")
    page.fill("#email", "admin@acme.com")
    page.fill("#password", "secret")
    

    sign_in_button_xpath = "/html/body/div[2]/div[1]/div/div/div[2]/form/button"

    page.locator(f"xpath={sign_in_button_xpath}").click()

    companies_xpath = "/html/body/div[2]/div[1]/div/aside/div/div[2]/nav/div[1]/div[1]/a[1]"
    page.locator(f"xpath={companies_xpath}").click()

    new_company_button_xpath = "/html/body/div[2]/div[1]/div/div/main/div/div/div/div[1]/div[1]/div[2]/a"
    page.locator(f"xpath={new_company_button_xpath}").click()

    fillin_company_name_xpath = "/html/body/div[2]/div[1]/div/div/main/div/div/div/div[2]/form/div[1]/div[1]/div[1]/div[2]/div/div[1]/input"
    page.locator(f"xpath={fillin_company_name_xpath}").fill("Test Company")
    create_new_company_xpath="/html/body/div[2]/div[1]/div/div/main/div/div/div/div[1]/div[2]/button"
    page.locator(f"xpath={create_new_company_xpath}").click()
    
    operation_successfull_xpath = "/html/body/div[2]/div[1]/div/div/main/div[1]/div/span[1]"
    expect(page.locator(f"xpath={operation_successfull_xpath}")).to_contain_text("Last operation:")