# pdf_parser.py

# Code from the first snippet
import tensorflow as tf
import tensorflow_hub as hub
import PyPDF2
import os
import pickle
import time
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain import PromptTemplate
import json


# pdf_parser.py

# Function from the first snippet
def extract_text_from_pdf(file_path):
    import tensorflow as tf
    import tensorflow_hub as hub
    import PyPDF2
    import os
    import pickle
    import time

    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(model_url)

    # Dictionary to store vectorized embeddings
    resume_embedding_dict = {}

    def extract_text_from_pdf(file_path):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

    def calculate_similarity(keyword_embedding, resume_embedding):
        similarity_score = tf.reduce_sum(tf.multiply(keyword_embedding, resume_embedding))
        return similarity_score

    # Accept the folder path containing resumes
    folder_path = "Multiple-Pdf-Resume-Analyzer/pdfs"

    # Load or create the vector embedding data
    vector_data_path = "vector_data.pickle"
    if os.path.exists(vector_data_path):
        with open(vector_data_path, "rb") as file:
            resume_embedding_dict = pickle.load(file)
    else:
        for filename in os.listdir(folder_path):
            resume_file = os.path.join(folder_path, filename)
            resume_text = extract_text_from_pdf(resume_file)
            resume_embedding = embed([resume_text])[0]
            resume_embedding_dict[resume_file] = resume_embedding
        # Save the vector embedding data
        with open(vector_data_path, "wb") as file:
            pickle.dump(resume_embedding_dict, file)

    keyword = input("Enter a keyword to search for: ")
    keyword_embedding = embed([keyword])[0]  # Convert keyword to a list
    # Save the keyword embedding
    keyword_embedding_path = "keyword_embedding.pickle"
    with open(keyword_embedding_path, "wb") as file:
        pickle.dump(keyword_embedding, file)

    # Process resumes and calculate similarity scores
    resumes = []
    for resume_file, resume_embedding in resume_embedding_dict.items():
        similarity_score = calculate_similarity([keyword_embedding],
                                                [resume_embedding])  # Convert keyword and resume_embedding to lists
        resumes.append((resume_file, similarity_score))
    # Sort the resumes based on similarity score
    resumes.sort(key=lambda x: x[1], reverse=True)
    start = time.time()

    # Display the ranked resumes
    print("Ranked Resumes:")
    for i, (resume_file, similarity_score) in enumerate(resumes):
        print(f"Rank {i + 1}: {resume_file} (Similarity Score: {similarity_score:.2f})")
    end = time.time()
    print(end - start)


# Function from the second snippet
def scan_resume(file):
    from dotenv import load_dotenv
    import streamlit as st
    from PyPDF2 import PdfReader
    import docx2txt
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    from langchain import PromptTemplate
    import json

    function_descriptions = [
        {
            "name": "scan_resume",
            "description": "Scans a resume and returns relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the person"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email of the person"
                    },
                    "phone": {
                        "type": "string",
                        "description": "Phone number of the person"
                    },
                    "education": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "school": {
                                    "type": "string",
                                    "description": "Name of the school"
                                },
                                "degree_or_certificate": {
                                    "type": "string",
                                    "description": "Degree or certificate"
                                },
                                "time_period": {
                                    "type": "string",
                                    "description": "Time period of education"
                                },
                            },
                        },
                        "description": "Education of the person",
                    },
                    "employment": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "company": {
                                    "type": "string",
                                    "description": "Name of the company"
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Title of the person"
                                },
                                "time_period": {
                                    "type": "string",
                                    "description": "Time period of employment"
                                },
                            },
                        },
                        "description": "Employment history of the person",
                    },
                    "skills": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Skills of the person"
                        },
                    },
                },
                "required": ["name", "email", "skills"]
            }
        }
    ]

    template = """/
    Scan the following resume and return the relevant details.
    If the data is missing just return N/A
    Resume: {resume}
    """

    def main():
        load_dotenv()

        llm = ChatOpenAI(model="gpt-4-0613")

        st.write("# Resume Scanner")

        st.write("### Upload Your Resume")

        status = st.empty()

        file = st.file_uploader("PDF, Word Doc", type=["pdf", "docx"])

        details = st.empty()

        if file is not None:
            with st.spinner("Scanning..."):
                text = ""
                if file.type == "application/pdf":
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()

                if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text += docx2txt.process(file)

                prompt = PromptTemplate.from_template(template)
                content = prompt.format(resume=text)

                response = llm.predict_messages(
                    [HumanMessage(content=content)],
                    functions=function_descriptions)

                data = json.loads(
                    response.additional_kwargs["function_call"]["arguments"])

            with details.container():
                st.write("## Details")
                st.write(f"Name: {data['name']}")
                st.write(f"Email: {data['email']}")
                st.write(f"Phone: {data['phone']}")
                st.write("Education:")
                for education in data['education']:
                    st.markdown(f"""
                        * {education['school']}
                            - {education['degree_or_certificate']}
                            - {education['time_period']}
                    """)
                st.write("Employment:")
                for employment in data['employment']:
                    st.markdown(f"""
                        * {employment['company']}
                            - {employment['title']}
                            - {employment['time_period']}
                    """)
                st.write("Skills:")
                for skill in data['skills']:
                    st.write(f" - {skill}")

            status = status.success("Resume Scanned Successfully")


# pdf_parser.py

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="PDF Parser")
    parser.add_argument("file_path", help="Path to the PDF file with CV")
    return parser.parse_args()


def main():
    args = parse_arguments()
    file_path = args.file_path

    # Call the functions based on your requirements
    text_from_pdf = extract_text_from_pdf(file_path)
    resume_data = scan_resume(file_path)

    # Output structured JSON
    print(json.dumps(resume_data, indent=2))


if __name__ == "__main__":
    main()
