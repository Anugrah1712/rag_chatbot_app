#Webscrape.py 

import os
import time
import asyncio
from playwright.async_api import async_playwright
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

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
    table_input = f"Table {table_index}:\n" + "\n".join(
        [", ".join(row) for row in table_data]
    )

    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [
                "Convert table to descriptive sentences. For example: The following information is for the customers under the age of 60 with a special period.\n"
                "● For customers with a tenure of 18 months, the interest rates are as follows: 7.80% per annum at maturity, 7.53% per annum for monthly interest, "
                "7.58% per annum for quarterly interest, 7.65% per annum for half-yearly interest, and 7.80% per annum for annual interest.\n"
                "● For a tenure of 22 months, the interest rates are 7.90% per annum at maturity, 7.63% per annum for monthly interest, "
                "7.68% per annum for quarterly interest, 7.75% per annum for half-yearly interest, and 7.90% per annum for annual interest.\n"
                "● For those with a 33-month tenure, the interest rates are 8.10% per annum at maturity, 7.81% per annum for monthly interest, "
                "7.87% per annum for quarterly interest, 7.94% per annum for half-yearly interest, and 8.10% per annum for annual interest.\n"
                "● Finally, for customers with a tenure of 44 months, the interest rates are 8.25% per annum at maturity, 7.95% per annum for monthly interest, "
                "8.01% per annum for quarterly interest, 8.09% per annum for half-yearly interest, and 8.25% per annum for annual interest."
            ]},
            {"role": "model", "parts": ["Please provide the table. I need the table's content to convert it into descriptive sentences.\n"]},
        ]
    )

    response = chat_session.send_message(table_input)
    return response.text

async def scrape_web_data():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to True to run in headless mode
        page = await browser.new_page()

        await page.goto(
            "https://www.bajajfinserv.in/investments/fixed-deposit-application-form?&utm_source=googleperformax_mktg&utm_medium=cpc&PartnerCode=76783&utm_campaign=DPPM_FD_OB_22072024_Vserv_PerfMax_Salaried&utm_term=&device=c&utm_location=9062096&utm_placement=&gad_source=1&gclid=EAIaIQobChMIo6z7toz3iQMV8uYWBR3D0jMGEAAYASAAEgKAdvD_BwE"
        )
        await page.wait_for_timeout(3000)

        tables = await page.query_selector_all("table")

        for i, table in enumerate(tables, start=1):
            print(f"Processing Table {i}:")
            
            rows = await table.query_selector_all("tr")
            table_data = []

            for row in rows:
                columns = await row.query_selector_all("td")
                column_text = [await column.inner_text() for column in columns]
                column_text = [text.strip() for text in column_text if text.strip()]
                if column_text:
                    table_data.append(column_text)

            if table_data:
                descriptive_sentences = convert_table_to_sentences_gemini(table_data, i)
                print(f"Descriptive Sentences for Table {i}:\n{descriptive_sentences}\n")

        body_text = await page.evaluate("document.body.innerText")
        faq_container = await page.query_selector(".faqs.aem-GridColumn.aem-GridColumn--default--12")

        if faq_container:
            while True:
                try:
                    show_more_button = await faq_container.query_selector(".accordion_toggle_show-more")
                    if show_more_button and await show_more_button.is_visible():
                        await show_more_button.click()
                        await page.wait_for_timeout(1000)
                    else:
                        break
                except Exception:
                    break

            toggle_buttons = await faq_container.query_selector_all(".accordion_toggle, .accordion_row")
            all_faqs = []
            for button in toggle_buttons:
                try:
                    await button.click()
                    await page.wait_for_timeout(1000)
                    expanded_content = await faq_container.query_selector_all(".accordion_body, .accordionbody_links, .aem-rte-content")
                    for content in expanded_content:
                        text = await content.inner_text()
                        text = text.strip()
                        if text and text not in [faq['answer'] for faq in all_faqs]:
                            question = await button.inner_text()
                            question = question.strip()
                            if question:
                                all_faqs.append({"question": question, "answer": text})
                except Exception as e:
                    print(f"Error interacting with button: {e}")

            print("\nExtracted FAQ Questions and Answers:")
            for i, faq in enumerate(all_faqs, start=1):
                print(f"Q: {faq['question']}\n   A: {faq['answer']}\n")

        print("Entire Page Content:")
        print(body_text)

        await browser.close()


    
#uvicorn main:app --reload