import json
import PyPDF2
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

OPENAI_API_KEY = 'my_key'

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, max_tokens=800)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def chunk_text(text, chunk_size=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    return splitter.split_text(text)

def create_vector_store(chunks):
    return FAISS.from_texts(chunks, embeddings)

def retrieve_relevant_text(vector_db, query, num_results=3):
    docs = vector_db.similarity_search(query, k=num_results)
    return "\n".join([doc.page_content for doc in docs])

def generate_questions_and_answers(text, question_type):
    if question_type == "Multiple Choice":
        prompt_template = """
        Generate a JSON response with diverse multiple-choice questions (MCQs) based on the given text.

        Response format:
        {{
            "questions":[
                {{
                    "question":"...",
                    "options":{{"A":"...","B":"...","C":"...","D":"..."}},
                    "answer":"C"
                }},
                {{
                    "question":"...",
                    "options":{{"A":"...","B":"...","C":"...","D":"..."}},
                    "answer":"A"
                }}
            ]
        }}

        Ensure:
        -Each MCQ tests factual recall or key concepts.
        -Each question has 4 answer choices (A,B,C,D).
        -One correct answer is explicitly labeled.
        -Questions do not overlap with short answer questions.

        Text: {text}
        JSON Output:
        """
    elif question_type == "Short Answer":
        prompt_template = """
        Generate a JSON response with short answer questions that require conceptual understanding.

        Response format:
        {{
            "questions":[
                {{"question":"...", "answer":"..."}},
                {{"question":"...", "answer":"..."}}
            ]
        }}

        Ensure:
        - Each question is different from MCQs.
        - Answers are 2-3 sentences long and provide context.
        - Use direct references from the provided text to support answers.

        Text: {text}
        JSON Output:
        """
    elif question_type == "Long Answer":
        prompt_template = """
        Generate a JSON response with long answer questions that require deeper explanations.

        Response format:
        {{
            "questions":[
                {{"question":"...", "answer":"..."}},
                {{"question":"...", "answer":"..."}}
            ]
        }}

        Ensure:
        - Each question requires explanation or example.
        - Answers are 5+ sentences long and include references from the text.
        - Provide detailed, well-structured answers with examples.

        Text: {text}
        JSON Output:
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response = llm.invoke(prompt.format(text=text))
    response_text = response.content if hasattr(response, 'content') else str(response)

    try:
        qa_pairs = json.loads(response_text.strip())
        if isinstance(qa_pairs, dict) and "questions" in qa_pairs:
            return qa_pairs["questions"]
        else:
            st.error("Unexpected JSON structure. Please try again.")
            return []
    except json.JSONDecodeError:
        st.error("Failed to parse JSON response from OpenAI. Please try again.")
        return []

st.title("Test Question Generator")

if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

uploaded_file = st.file_uploader("Upload a PDF of notes or textbooks", type=["pdf"])
question_type = st.selectbox("Select Question Type", ["Multiple Choice", "Short Answer", "Long Answer"])
display_question = st.button("Generate Questions")

if uploaded_file and display_question:
    with st.spinner("Processing..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        text_chunks = chunk_text(extracted_text)
        vector_db = create_vector_store(text_chunks)
        relevant_text = retrieve_relevant_text(vector_db, question_type)
        qa_pairs = generate_questions_and_answers(relevant_text, question_type)

    if qa_pairs:
        st.session_state.qa_pairs = qa_pairs

qa_pairs = st.session_state.qa_pairs

if qa_pairs:
    st.subheader("Generated Questions")
    for idx, qa in enumerate(qa_pairs):
        st.write(f"**Q{idx+1}:** {qa['question']}")
        if question_type == "Multiple Choice" and "options" in qa:
            st.write(f"A) {qa['options']['A']}")
            st.write(f"B) {qa['options']['B']}")
            st.write(f"C) {qa['options']['C']}")
            st.write(f"D) {qa['options']['D']}")

    if st.button("Check Answers"):
        st.subheader("Answers")
        for idx, qa in enumerate(qa_pairs):
            if question_type == "Multiple Choice":
                st.write(f"**A{idx+1}:** Correct Answer **{qa['answer']}**")
            else:
                st.write(f"**A{idx+1}:** {qa['answer']}")
