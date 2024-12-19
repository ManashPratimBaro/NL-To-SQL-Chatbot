# **AI-Powered Natural Language to SQL Query Generator**
Overview
The AI-Powered Natural Language to SQL Query Generator is a Natural Language Processing (NLP) project that aims to simplify database querying by translating plain English queries into SQL commands. This project enhances accessibility and user experience, making data interaction more intuitive across various domains, including business intelligence and academic research.

## **Table of Contents**
- Features
- Technologies Used
- Installation
- Usage
- Project Structure
- Contributing

# Features
Convert natural language queries into SQL commands.
User-friendly interface for easy interaction.
Integration with MySQL databases for real-time query execution.
Supports complex queries and provides accurate SQL outputs.
Built using advanced NLP techniques and transformer models.

# Technologies Used
- Python: Primary programming language for development.
- FastAPI: For creating API endpoints to handle natural language queries and SQL execution.
- PyTorch: For building and deploying deep learning models.
- Transformers Library: For utilizing transformer-based models like T5 for NLP tasks.
- MySQL: For database management and query execution.
- HTML, CSS, JavaScript: For frontend development and user interface design.

# Installation
To set up the project locally, follow these steps:

Clone the repository:
git clone https://github.com/ManashPratimBaro/NL-to-SQL-Chatbot.git
cd nlp-to-sql-generator

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:
pip install -r requirements.txt
Set up your MySQL database and update the database connection details in the configuration file.

# Usage
Start the FastAPI server:
uvicorn main:app --reload
Open your web browser and navigate to http://127.0.0.1:8000.
Enter your natural language query in the input field and submit to receive the corresponding SQL output.

## **Project Structure**
```plaintext
nlp-to-sql-generator/
├── main.py                # Main application file for FastAPI
├── requirements.txt       # Python dependencies
├── templates/             # HTML templates for the frontend
│   └── index.html         # Main HTML file
├── static/                # Static files for styling and frontend scripts
│   ├── styles.css         # CSS file for styling
│   └── script.js          # JavaScript file for frontend interactivity
├── models/                # Directory for machine learning models
│   ├── model.py           # Code for the ML model
│   └── tokenizer.py       # Tokenizer utilities
└── config/                # Configuration files
    └── db_config.py       # MySQL database configuration
```


# Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request.

Fork the repository.
Create your feature branch:
git checkout -b feature/YourFeature

Commit your changes:
git commit -m 'Add some feature'

Push to the branch:
git push origin feature/YourFeature
Open a pull request.
