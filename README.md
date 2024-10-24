# Parsely Tool

Parsely is a flexible and customizable tool designed to parse, transform, store, and query data. It is suitable for data engineers looking to quickly test and demo data output after a Retrieve-Augment-Generate (RAG) data pull request.

## Features

- **Modular Design**: Separate modules for parsing, storage, querying, and utilities.
- **Asynchronous Parsing**: Efficiently parse multiple files asynchronously.
- **Storage with Qdrant**: Store embeddings and metadata in a vector database.
- **Query Engine**: Perform similarity searches and retrieve relevant documents.
- **Configurable**: Easily set up configurations using environment variables or configuration files.
- **Streamlit Demo App**: A web interface for uploading files and querying data.

## Installation

```bash
pip install -r requirements.txt

Usage
Example Script
Run the example usage script:

```bash
Copy code
python examples/example_usage.py
Streamlit App
Launch the demo app:

```bash
Copy code
streamlit run examples/demo_app.py
Configuration
Set up the necessary environment variables:

OPENAI_API_KEY
QDRANT_API_KEY
QDRANT_URL
PARSE_API_URL
COHERE_API_KEY
Testing
Run tests using:

```bash
Copy code
python -m unittest discover tests
License
MIT License

```arduino
Copy code

This README provides an overview and instructions for using the `parsely_tool`.

---

## setup.py

```python
# setup.py

from setuptools import setup, find_packages

setup(
    name='parsely_tool',
    version='0.1.0',
    description='A flexible tool for parsing, transforming, storing, and querying data.',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'aiofiles',
        'aiohttp',
        'pandas',
        'qdrant-client',
        'openai',
        'streamlit',
        'asyncio',
        'uvloop',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)
This setup script allows you to install the parsely_tool package using pip.

#   p a r s e l y _ t o o l  
 