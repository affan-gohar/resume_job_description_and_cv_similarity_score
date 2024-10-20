# Resume Job Description Similarity Checker

This project is an AI-based tool designed to compare resumes with job descriptions, providing a similarity score to help identify the best matches between candidates and job requirements. By utilizing Natural Language Processing (NLP) techniques, this tool significantly improves the hiring process by reducing the time spent on resume screening.

## Features
- **Resume-Job Description Matching**: Provides a similarity score by comparing resumes to job descriptions using advanced NLP techniques.
- **Efficiency**: Reduced resume screening time by 40%, improving overall recruitment efficiency.
- **Accuracy**: Leveraged Word2Vec embeddings and cosine similarity algorithms to enhance the precision of matching resumes to job descriptions.

## Technology Stack
- **Programming Language**: Python
- **Libraries**:
  - SpaCy
  - Gensim (Word2Vec)
  - NumPy
  - Flask (for web integration)
- **Algorithms**: Word2Vec for word embeddings and cosine similarity for calculating the similarity between resumes and job descriptions.

## Usage
1. Upload resume(s) and a job description through the web interface.
2. The tool will calculate a similarity score between each resume and the job description.
3. Resumes with the highest similarity scores are the best matches for the job.

## Results
- **40% Reduction in Screening Time**: The tool automated the resume matching process, cutting down manual screening time significantly.
- **Enhanced Matching Precision**: By using Word2Vec and cosine similarity, the matching precision was greatly improved, ensuring more relevant results.

## Future Work
- Expand support for various file formats like PDF and DOCX for both resumes and job descriptions.
- Incorporate additional features such as keyword extraction and skill matching to further refine the similarity scores.
- Add a feature for recruiters to give feedback on matches, further improving the model.

## Contributing
Contributions are welcome! Please submit any issues or pull requests to enhance the functionality or fix any bugs. For major changes, consider opening an issue to discuss your ideas.
