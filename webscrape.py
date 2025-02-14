from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
service = Service(r"D:\chromedriver-win64\chromedriver-win64\chromedriver.exe")
driver = webdriver.Chrome(service=service)

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
    # Format the table data as input for the AI model
    table_input = f"Table {table_index}:\n" + "\n".join(
        [", ".join(row) for row in table_data]
    )

    # Start a chat session
    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [
                "Convert table to descriptive sentences.For example:The following information is for the customers under the age of 60 with a special period.● For customers with a tenure of 18 months, the interest rates are as follows: 7.80% per annum at maturity, 7.53% per annum for monthly interest, 7.58% per annum for quarterly interest, 7.65% per annum for half-yearly interest, and 7.80% per annum for annual interest.● For a tenure of 22 months, the interest rates are 7.90% per annum at maturity, 7.63% per annum for monthly interest, 7.68% per annum for quarterly interest, 7.75% per annum for half-yearly interest, and 7.90% per annum for annual interest.● For those with a 33-month tenure, the interest rates are 8.10% per annum at maturity, 7.81% per annum for monthly interest, 7.87% per annum for quarterly interest, 7.94% per annum for half-yearly interest, and 8.10% per annum for annual interest.● Finally, for customers with a tenure of 44 months, the interest rates are 8.25% per annum at maturity, 7.95% per annum for monthly interest, 8.01% per annum for quarterly interest, 8.09% per annum for half-yearly interest, and 8.25% per annum for annual interest."]},
            {
                "role": "model",
                "parts": [
                    "Please provide the table. I need the table's content to convert it into descriptive sentences.\n"
                ],
            },
        ]
    )

    # Send the formatted table data to the model
    response = chat_session.send_message(table_input)
    return response.text


try:
    driver.get(
        "https://www.bajajfinserv.in/investments/fixed-deposit-application-form?&utm_source=googleperformax_mktg&utm_medium=cpc&PartnerCode=76783&utm_campaign=DPPM_FD_OB_22072024_Vserv_PerfMax_Salaried&utm_term=&device=c&utm_location=9062096&utm_placement=&gad_source=1&gclid=EAIaIQobChMIo6z7toz3iQMV8uYWBR3D0jMGEAAYASAAEgKAdvD_BwE")
    time.sleep(3)
    tables = driver.find_elements(By.TAG_NAME, "table")

    for i, table in enumerate(tables, start=1):
        print(f"Processing Table {i}:")

        # Extract table rows
        rows = table.find_elements(By.TAG_NAME, "tr")
        table_data = []

        for row in rows:
            columns = row.find_elements(By.TAG_NAME, "td")
            column_text = [column.text.strip() for column in columns]
            if column_text:  # Skip empty rows
                table_data.append(column_text)

        # Use Gemini AI to convert table data to sentences
        if table_data:
            descriptive_sentences = convert_table_to_sentences_gemini(table_data, i)
            print(f"Descriptive Sentences for Table {i}:\n{descriptive_sentences}\n")

    body_text = driver.find_element(By.TAG_NAME, "body").text
    faq_container = driver.find_element(By.CSS_SELECTOR, ".faqs.aem-GridColumn.aem-GridColumn--default--12")

    while True:
        try:
            show_more_button = faq_container.find_element(By.CSS_SELECTOR, ".accordion_toggle_show-more")
            if show_more_button.is_displayed():
                show_more_button.click()
                time.sleep(1)
            else:
                break
        except Exception:
            break

    toggle_buttons = faq_container.find_elements(By.CSS_SELECTOR, ".accordion_toggle, .accordion_row")
    all_faqs = []
    for button in toggle_buttons:
        try:
            button.click()
            time.sleep(1)
            expanded_content = faq_container.find_elements(By.CSS_SELECTOR,
                                                           ".accordion_body, .accordionbody_links, .aem-rte-content")
            for content in expanded_content:
                text = content.text.strip()
                if text and text not in [faq['answer'] for faq in all_faqs]:
                    question = button.text.strip()
                    if question:
                        all_faqs.append({"question": question, "answer": text})

        except Exception as e:
            print(f"Error interacting with button: {e}")

    print("Entire Page Content:")
    print(body_text)

    print("\nExtracted FAQ Questions and Answers:")
    for i, faq in enumerate(all_faqs, start=1):
        print(f"Q: {faq['question']}\n   ")

finally:
    driver.quit()



