import os
import tempfile
from flask import Flask, render_template, request
import re
import docx2txt
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np

app = Flask(__name__, static_url_path='/static')

# extracing text from docx
def extract_text_from_word(file_path):
    return docx2txt.process(file_path)

# extract text from pdf
def extract_text_from_pdf(file_path):
    return extract_text(file_path)

# extracting job description or resume text
def extract_file_text(file, file_type):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        file.save(temp_file.name)
        if file_type == 'word':
            return extract_text_from_word(temp_file.name)
        elif file_type == 'pdf':
            return extract_text_from_pdf(temp_file.name)
        else:
            print("Unsupported file type!")
            exit()

# tokenization using regex
def tokenize(text):
    text = text.lower()  #  lowercase conversion
    return re.findall(r'\b\w+\b', text)

#  calculating average Word2Vec vector for a section of text
def average_word2vec_vector(model, text):
    tokens = tokenize(text)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# calculaitng similarity score using w2c
def calculate_similarity_with_word2vec(model, text1, text2):
    vector1 = average_word2vec_vector(model, text1)
    vector2 = average_word2vec_vector(model, text2)
    return cosine_similarity([vector1], [vector2])[0][0]

# Route for handling the form submission and processing the resumes and job descriptions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        choice = request.form['choice']  # User's choice: 'cv' or 'job'

        if choice == 'cv':
            job_description_file = request.files['job_description']
            job_type = request.form['job_type']
            resume_files = request.files.getlist('resumes')
            resume_type = request.form['resume_type']

            # Extract job des text
            job_description_text = extract_file_text(job_description_file, job_type)

            resume_texts = []
            resume_names = []
            for resume_file in resume_files:
                resume_text = extract_file_text(resume_file, resume_type)
                resume_texts.append(resume_text)
                resume_names.append(resume_file.filename)  # Store resume filename

            # Tokenize and train Word2Vec model
            all_text = job_description_text + " " + " ".join(resume_texts)
            tokenized_text = [tokenize(all_text)]
            model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

            # Calculate similarity between job description and each resume
            similarities = []
            for resume_text in resume_texts:
                similarity = calculate_similarity_with_word2vec(model, resume_text, job_description_text)
                similarities.append(similarity)

            # Find the best matching resume
            best_resume_index = np.argmax(similarities)
            best_resume_name = resume_names[best_resume_index]
            best_similarity = similarities[best_resume_index]

            # Format the result for output
            result = [{'filename': resume_names[i], 'similarity_score': similarities[i]} for i in range(len(similarities))]
            best_match = {'filename': best_resume_name, 'similarity_score': best_similarity}

        elif choice == 'job':
            resume_file = request.files['resume']
            resume_type = request.form['resume_type']
            job_description_files = request.files.getlist('job_descriptions')
            job_type = request.form['job_type']

            # Extract resume text
            resume_text = extract_file_text(resume_file, resume_type)

            job_texts = []
            job_names = []
            for job_file in job_description_files:
                job_text = extract_file_text(job_file, job_type)
                job_texts.append(job_text)
                job_names.append(job_file.filename)  # Store job description filename

            # Combine all job descriptions for training the Word2Vec model
            all_text = resume_text + " " + " ".join(job_texts)
            tokenized_text = [tokenize(all_text)]
            model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

            # Calculate similarity between resume and each job description
            similarities = []
            for job_text in job_texts:
                similarity = calculate_similarity_with_word2vec(model, resume_text, job_text)
                similarities.append(similarity)

            # Find the best matching job description
            best_job_index = np.argmax(similarities)
            best_job_name = job_names[best_job_index]
            best_similarity = similarities[best_job_index]

            # Format the result for output
            result = [{'filename': job_names[i], 'similarity_score': similarities[i]} for i in range(len(similarities))]
            best_match = {'filename': best_job_name, 'similarity_score': best_similarity}

        return render_template('index.html', result=result, best_match=best_match)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
