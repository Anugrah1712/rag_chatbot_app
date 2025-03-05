from playwright.sync_api import sync_playwright
import os
import time
import google.generativeai as genai

# Configure Gemini AI (Ensure API key is set)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing! Set it in environment variables.")

genai.configure(api_key=api_key)

# Configure Gemini AI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

def convert_table_to_sentences_gemini(table_data, table_index):
    """Converts table data into descriptive sentences using Gemini Generative AI."""
    table_input = f"Table {table_index}:\n" + "\n".join([", ".join(row) for row in table_data])

    chat_session = model.start_chat(history=[
        {"role": "user", "parts": [
            "Convert table to descriptive sentences. Example format:\n"
            "For customers under 60 with a special period:\n"
            "● 18 months: 7.80% per annum at maturity, 7.53% per annum for monthly interest...\n"
            "Now, process the table I provide."
        ]},
        {"role": "model", "parts": ["Please provide the table content."]},
    ])

    response = chat_session.send_message(table_input)
    return response.text

def scrape_data(url="https://www.bajajfinserv.in/investments/fixed-deposit-application-form"):
    """Scrapes the given URL for tables and FAQs using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Change to False for debugging
        page = browser.new_page()

        try:
            # Open webpage
            page.goto(url, timeout=60000)
            time.sleep(3)

            # Extract tables
            tables = page.locator("table").all()
            extracted_data = {"tables": [], "faqs": []}

            for i, table in enumerate(tables, start=1):
                print(f"Processing Table {i}:")
                rows = table.locator("tr").all()
                table_data = [[col.inner_text().strip() for col in row.locator("td").all()] for row in rows if row.inner_text()]

                if table_data:
                    descriptive_sentences = convert_table_to_sentences_gemini(table_data, i)
                    extracted_data["tables"].append({"index": i, "sentences": descriptive_sentences})

            # Extract FAQ Section
            faq_container = page.locator(".faqs.aem-GridColumn.aem-GridColumn--default--12")

            # Click "Show More" buttons if available
            while True:
                try:
                    show_more_button = faq_container.locator(".accordion_toggle_show-more")
                    if show_more_button.is_visible():
                        show_more_button.click()
                        time.sleep(1)
                    else:
                        break
                except:
                    break

            toggle_buttons = faq_container.locator(".accordion_toggle, .accordion_row").all()

            for button in toggle_buttons:
                try:
                    button.click()
                    time.sleep(1)

                    expanded_content = faq_container.locator(".accordion_body, .accordionbody_links, .aem-rte-content").all()
                    for content in expanded_content:
                        text = content.inner_text().strip()
                        if text and text not in [faq['answer'] for faq in extracted_data["faqs"]]:
                            question = button.inner_text().strip()
                            if question:
                                extracted_data["faqs"].append({"question": question, "answer": text})

                except Exception as e:
                    print(f"Error interacting with button: {e}")

            return extracted_data

        finally:
            browser.close()
