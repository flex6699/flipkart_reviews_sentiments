import re
import time
import streamlit as st
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from PIL import Image
import numpy as np
import os





# Hugging Face setup
api_token = st.secrets["secrets"]["HUGGINGFACEHUB_API_TOKEN"]
# Retrieve Hugging Face API token from environment variable

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=api_token, temperature=0.5)
template = """
    You are an AI assistant that follows instructions extremely well. Please be truthful and give direct answers.

    Your task is to perform sentiment analysis on the following text. Identify the overall sentiment of the text as either "positive," "negative," or "neutral."

    Here is the text for analysis:

    {query}

    Please provide your analysis in one word: "positive" or "negative"  I need answer only in negative, positive 

    Thank you.
    """
prompt = PromptTemplate.from_template(template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def clean_review(review):
    cleaned_review = re.sub(r'[^\w\s]', '', review.replace('\n', ''))
    index_read_more = cleaned_review.find("READ MORE")
    if index_read_more != -1:
        cleaned_review = cleaned_review[:index_read_more].strip()
    return cleaned_review

def scrape_reviews(url, max_page):
    total_reviews = []
    for i in range(1, max_page + 1):
        page_url = f"{url}&page={i}"
        driver = None
    try:
        # Using on Local
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1200')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                                  options=options)
        st.write(f"DEBUG:DRIVER:{driver}")
        driver.get(url)
        time.sleep(5)
        html_doc = driver.page_source
        driver.quit()
        soup = BeautifulSoup(html_doc, "html.parser")
        
        reviews_all = soup.find_all(class_='ZmyHeo')
        print(reviews_all)
        if not reviews_all:
            print(f"No reviews found on page {i}")
              
        reviews = [clean_review(review.text) for review in reviews_all]
        total_reviews.extend(reviews)
        return total_reviews
    except Exception as e:
        st.write(f"DEBUG:INIT_DRIVER:ERROR:{e}")
    finally:
        if driver is not None: driver.quit()
    
       

def llmResponse(query):
    response = llm_chain.run(query)
    return response.split("\n")[-1].strip()

# Streamlit app
st.title("Flipkart Review Sentiment Analysis")

url = st.text_input("Enter the Flipkart product review URL:")
max_page = st.number_input("Enter the number of pages to scrape:", min_value=1, max_value=10, value=2)

if st.button("Scrape and Analyze Reviews"):
    with st.spinner("Scraping reviews..."):
        reviews = scrape_reviews(url, max_page)
    
    with st.spinner("Analyzing sentiment..."):
        results = [(review, llmResponse(review)) for review in reviews]
    print(results)
    st.success("Analysis complete!")
    for review, sentiment in results:
        if sentiment == "Positive":
            st.markdown(f'<div style="background-color: #FFFF00; padding: 10px; margin: 10px; border-radius: 5px;">\
                         <span title="Positive" style="display: inline-block; width: 100%; height: 100%; cursor: help;">{review}</span></div>', unsafe_allow_html=True)
        elif sentiment == "Negative":
            st.markdown(f'<div style="background-color: #FF0000; padding: 10px; margin: 10px; border-radius: 5px;">\
                         <span title="Negative" style="display: inline-block; width: 100%; height: 100%; cursor: help;">{review}</span></div>', unsafe_allow_html=True)

# Quit the driver after scraping all pages

