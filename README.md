# Low-Code Text Analysis and Visualization Tool

This is a low-code tool built with Python and Streamlit for performing various text analysis tasks. It provides a simple web-based interface for users to upload text files and visualize the results of different analyses without writing any code.

## Features

The tool currently supports the following analysis and visualization methods:

*   **Word Frequency Analysis**: Calculates and displays the most frequent words in the text, along with a bar chart visualization.
*   **TF-IDF Keyword Extraction**: Identifies the most important keywords in the text using the TF-IDF algorithm.
*   **LDA Topic Modeling**: Discovers abstract topics from the text using Latent Dirichlet Allocation.
*   **Named Entity Recognition (NER)**: Identifies and categorizes named entities such as persons, organizations, and locations.
*   **Entity Co-occurrence Network**: Visualizes the relationships between named entities that appear in the same sentences.
*   **Word Cloud Visualization**: Creates a visual representation of the most prominent words in the text.

## Setup and Installation

### 1. Prerequisites

*   Python 3.8+
*   pip (Python package installer)

### 2. Clone the Repository

```bash
git clone https://github.com/battlewin3/SJWL-TextAnalisy.git
cd SJWL-TextAnalisy
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Download Language Model

The Named Entity Recognition (NER) and network analysis features require a pre-trained language model from `spaCy`. Download the recommended Chinese model using the following command:

```bash
python -m spacy download zh_core_web_sm
```

### 5. Font Installation (Important for Visualizations)

The Word Cloud and Network Graph visualizations require a font that supports Chinese characters. The application is currently configured to use `SimHei`.

Please ensure you have this font or a similar Chinese font installed on your system. If you use a different font, you may need to update the font path in `src/processing.py` (for word clouds) and `app.py` (for network graphs).

## How to Run the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
python -m streamlit run app.py
```

This will start a local web server and open the application in your default web browser. You can then upload a `.txt` file and select an analysis method from the sidebar to begin.

---
*This tool was developed iteratively with the assistance of a large language model.*
