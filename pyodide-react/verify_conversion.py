from playwright.sync_api import Page, expect

def test_conversion(page: Page):
    """
    This test verifies that the pyodide-react application can successfully
    convert a .drawio file and a .csv file into RML.
    """
    # 1. Arrange: Go to the application's page.
    page.goto("http://localhost:8000")

    # 2. Act: Upload the files and click the convert button.
    drawio_file_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
    csv_file_path = "gbad/mapping/source/generic.csv"

    page.locator('input[type="file"]').first.set_input_files(drawio_file_path)
    page.locator('input[type="file"]').last.set_input_files(csv_file_path)

    page.get_by_role("button", name="Convert").click()

    # 3. Assert: Confirm that the conversion was successful.
    output_element = page.locator("pre")
    expect(output_element).to_be_visible(timeout=60000) # 60 seconds timeout for pyodide loading and processing
    expect(output_element).to_contain_text("@base <https://data.archives.gov.on.test.gbad.ca/>")

    # 4. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="/app/pyodide-react/verification.png")
