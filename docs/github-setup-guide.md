# GitHub Setup Guide for Naval Ravikant Chatbot

This guide will walk you through setting up version control for your Naval Ravikant Chatbot project using GitHub.

## Initial Setup

### 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in to your account (or create one if needed)
2. Click the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "naval-ravikant-chatbot")
4. Add a description (e.g., "A chatbot based on The Almanack of Naval Ravikant using GPT-4")
5. Choose whether to make it public or private
6. Select "Add a README file"
7. Select "Add .gitignore" and choose "Python" from the template dropdown
8. Choose a license if desired
9. Click "Create repository"

### 2. Clone the Repository to Your Local Machine

Open your terminal/command prompt and run:

```bash
git clone https://github.com/your-username/naval-ravikant-chatbot.git
cd naval-ravikant-chatbot
```

## Project Structure

Create a well-organized directory structure for your project:

```bash
# Create directories
mkdir -p data
mkdir -p notebooks
mkdir -p src/utils
mkdir -p models
mkdir -p docs
```

### Recommended Project Layout

```
naval-ravikant-chatbot/
├── data/                  # Store the book PDF and processed data
├── notebooks/             # Jupyter notebooks for development and demonstration
├── src/                   # Source code
│   ├── utils/             # Utility functions
│   ├── ingestion.py       # Book ingestion and processing code
│   ├── embeddings.py      # Vector embedding creation code
│   ├── retrieval.py       # Retrieval system code
│   └── conversation.py    # Conversation management code
├── models/                # Store saved models or vector databases
├── docs/                  # Documentation
│   ├── Naval-Ravikant-Book-Chat-Bot-Project-Documentation.md
│   └── Naval-Ravikant-Chatbot-Implementation-Guide.md
├── requirements.txt       # Project dependencies
├── .gitignore             # Specifies files to ignore in version control
├── README.md              # Project overview
└── naval_chatbot.ipynb    # Main notebook for the chatbot
```

## Setting Up .gitignore

Update your .gitignore file to exclude sensitive information and large files:

```
# API keys and sensitive information
.env
**/api_key*

# Large data files
*.pdf
*.pkl
*.bin
vector_store/

# Python cache files
__pycache__/
*.py[cod]
*$py.class
.ipynb_checkpoints/

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual environments
venv/
env/
ENV/
```

## README Setup

Update your README.md with essential information:

```markdown
# Naval Ravikant Chatbot

A Jupyter notebook-based chatbot that lets you have conversations with a virtual Naval Ravikant based on "The Almanack of Naval Ravikant."

## Features

- Interactive chat interface in Jupyter notebooks
- Uses GPT-4 API and Retrieval-Augmented Generation
- Contextual responses based on Naval's book
- Preserves Naval's communication style and philosophy

## Setup and Installation

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Add your OpenAI API key to a `.env` file
4. Place "The Almanack of Naval Ravikant" PDF in the `data/` directory
5. Run the main notebook: `jupyter notebook naval_chatbot.ipynb`

## Project Structure

[Brief description of project structure]

## Documentation

See the `docs/` directory for detailed documentation:
- [Project Documentation](docs/Naval-Ravikant-Book-Chat-Bot-Project-Documentation.md)
- [Implementation Guide](docs/Naval-Ravikant-Chatbot-Implementation-Guide.md)

## License

[Your license here]
```

## Managing API Keys Securely

Create a `.env` file for storing sensitive information (make sure it's in your .gitignore):

```
OPENAI_API_KEY=your-api-key-here
```

Add code to load this in your notebooks:

```python
from dotenv import load_dotenv
import os

# Load API keys from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

## Workflow for Development

### 1. Initial Commit

After setting up the structure:

```bash
git add .
git commit -m "Initial project setup and documentation"
git push origin main
```

### 2. Regular Development Workflow

1. Create a new branch for each feature:
   ```bash
   git checkout -b feature/book-ingestion
   ```

2. Make changes and commit frequently:
   ```bash
   git add .
   git commit -m "Implement PDF text extraction"
   ```

3. Push changes to GitHub:
   ```bash
   git push origin feature/book-ingestion
   ```

4. Create a pull request on GitHub to merge into main

5. After merging, pull the updated main branch:
   ```bash
   git checkout main
   git pull
   ```

## Recommended Development Approach

1. Start with a development notebook for each component:
   - `notebooks/1_book_ingestion.ipynb`
   - `notebooks/2_embedding_creation.ipynb`
   - etc.

2. Once code is working, refactor into proper Python modules in the `src/` directory

3. Create a final integration notebook that imports from these modules

## Requirements File

Create a `requirements.txt` file with all dependencies:

```
openai==0.28.0
langchain==0.0.267
pypdf==3.15.1
transformers==4.31.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
ipywidgets==8.0.7
jupyter==1.0.0
python-dotenv==1.0.0
numpy==1.25.2
pandas==2.0.3
```

## Git Commit Best Practices

- Make atomic commits (one logical change per commit)
- Write meaningful commit messages
- Commit often
- Push regularly to avoid losing work

## Working with Large Files

If your book PDF or vector database is large (>100MB), consider using Git LFS (Large File Storage):

1. Install Git LFS: https://git-lfs.com/
2. Set up tracking for large file types:
   ```bash
   git lfs install
   git lfs track "*.pdf" "*.pkl"
   git add .gitattributes
   ```

## Managing Jupyter Notebooks in Git

To keep notebook outputs from cluttering your git history:

1. Install nbstripout:
   ```bash
   pip install nbstripout
   ```

2. Set up for your repository:
   ```bash
   nbstripout --install
   ```

This will automatically remove output cells when committing notebooks.
