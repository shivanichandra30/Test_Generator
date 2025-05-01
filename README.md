# Test_Generator
This is a Streamlit-based web application that generates educational questions from uploaded PDF notes or textbooks. Leveraging LangChain, OpenAI GPT-3.5, and FAISS, the app can create:

# 1. Multiple Choice Questions (MCQs)

# 2. Short Answer Questions

# 3. Long/Essay Answer Questions

It provides an intuitive interface for educators and learners to transform learning material into practice questions effortlessly.

## Features
1. Upload a PDF and automatically extract text

2. Use embeddings and vector search to find relevant content

3. Generate MCQs, short, and long answer questions using GPT-3.5

4. Interactive question viewing with answer checking

5. Built with LangChain, FAISS, OpenAI, and Streamlit

##  How to Run the App

Follow these steps to set up and run the Test Question Generator locally.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/test-question-generator.git
cd test-question-generator
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Set Your OpenAI API Key
Create a .env file in the root directory of the project and add:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
### 5. Run the Streamlit App
```bash
streamlit run app.py
```
### 6. Use the App
Upload a PDF file (e.g., notes, textbook)

Select the question type (Multiple Choice, Short Answer, or Long Answer)

Click Generate Questions

View the generated questions and answers interactively
