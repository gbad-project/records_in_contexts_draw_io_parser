from playwright.sync_api import Page, expect

def test_conversion(page: Page):
    """
    This test verifies that the pyodide-react application can successfully
    convert a .drawio file and a .csv file into RML.
    """
    # 1. Arrange: Go to the application's page.
    page.goto(
        "http://localhost:3000",
        wait_until="commit",
        timeout=120000,
    )

    # Wait for Pyodide and its packages to finish loading. During
    # initialization the main thread is busy, which prevents Playwright from
    # interacting with the file inputs and leads to timeouts. The "Convert"
    # button only becomes available once the page is idle, so wait for it
    # before proceeding.
    page.get_by_role("button", name="Convert").wait_for(timeout=120000)

    # 2. Act: Upload the files and click the convert button.
    drawio_file_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
    csv_file_path = "gbad/mapping/source/generic.csv"

    page.locator('input[type="file"]').first.set_input_files(
        drawio_file_path, timeout=300000
    )
    page.locator('input[type="file"]').last.set_input_files(
        csv_file_path, timeout=300000
    )

    page.get_by_role("button", name="Convert").click()

    # 3. Assert: Confirm that the conversion was successful.
    output_element = page.locator("pre")
    expect(output_element).to_be_visible(timeout=60000) # 60 seconds timeout for pyodide loading and processing
    expect(output_element).to_contain_text("@base <https://data.archives.gov.on.test.gbad.ca/>")

    # 4. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="./verification.png")
