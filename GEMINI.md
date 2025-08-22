## Project Overview

This is a low-code text analysis and visualization tool built with Python and Streamlit. It provides a web-based interface for users to upload text files and perform various analyses, such as word frequency, TF-IDF keyword extraction, LDA topic modeling, named entity recognition (NER), and entity co-occurrence network visualization.

The main application logic is in `app.py`, which uses the Streamlit framework to create the user interface. The core text processing functions are located in `src/processing.py`. The tool is designed to work with Chinese text and requires the `zh_core_web_sm` spaCy model for NER tasks and the `SimHei` font for visualizations.

## Building and Running

### 1. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Download Language Model

Download the necessary spaCy model for Chinese language processing:

```bash
python -m spacy download zh_core_web_sm
```

### 3. Run the Application

Start the Streamlit application with the following command:

```bash
streamlit run app.py
```

## Development Conventions

The project follows a simple structure with the main application in the root directory (`app.py`) and the core processing logic in the `src` directory (`src/processing.py`).

- All text processing functions are located in `src/processing.py`.
- The Streamlit UI and application flow are managed in `app.py`.
- The application is configured to use the `SimHei` font for word clouds and network graphs. If this font is not available, the font path in `src/processing.py` and `app.py` may need to be updated.
