# setup.py

from setuptools import setup, find_packages

setup(
    name='parsely_tool',
    version='0.1.0',
    description='A flexible tool for parsing, transforming, storing, and querying data.',
    author='Homen Shum',
    author_email='hshum2018@gmail.com',
    packages=find_packages(),
    install_requires=[
        'aiofiles',
        'aiohttp',
        'biopython',
        'cohere',
        'deep-translator',
        'easyocr',
        'fastapi',
        'fastembed',
        'frontend',
        'git+https://github.com/openai/CLIP.git',
        'httpx-oauth',
        'icecream',
        'instructor',
        'IPython',
        'llama-hub',
        'llama-index',
        'llama-index-embeddings-azure-openai',
        'llama-index-embeddings-huggingface-optimum',
        'llama-index-llms-azure-openai',
        'llama-index-llms-openai',
        'llama-index-retrievers-bm25',
        'llama-index-indices-managed-vectara',
        'llama-index-vector-stores-qdrant',
        'llama-parse',
        'llmsherpa',
        'nest_asyncio',
        'numpy',
        'onnxruntime',
        'openai',
        'openpyxl',
        'optimum',
        'optimum[exporters]',
        'optimum[onnxruntime]',
        'pandas',
        'pdfminer',
        'pillow',
        'portalocker',
        'psutil',
        'pydantic',
        'python-dotenv',
        'qdrant_client',
        'qdrant-client[fastembed]',
        'rank-bm25',
        'scikit-learn',
        'sentence-transformers',
        'spider-client',
        'streamlit',
        'streamlit-antd-components',
        'streamlit-authenticator',
        'streamlit-camera-input-live',
        'streamlit-card',
        'streamlit-carousel',
        'streamlit-extras',
        'streamlit-searchbox',
        'streamlit-webrtc',
        'transformers',
        'unstructured',
        'unstructured[docx]',
        'unstructured[pdf]',
        'unstructured_inference',
        'uvicorn',
        'xlsxwriter',
    ],
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)