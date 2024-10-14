# parsely_tool/parsers.py

import asyncio
import aiofiles
import aiohttp
import pandas as pd
import os
import tempfile
import logging
from typing import List, Union
from io import BytesIO
from datetime import datetime
import json
from openai import OpenAI
from llama_index.core.schema import Document

class Parser:
    def __init__(self, config):
        self.config = config
        self.supported_extensions = [
            ".docx", ".doc", ".odt", ".pptx", ".ppt", ".xlsx", ".csv", ".tsv",
            ".eml", ".msg", ".rtf", ".epub", ".html", ".xml", ".pdf",
            ".png", ".jpg", ".jpeg", ".txt"
        ]
        self.parsed_documents = []  # Change to list of Document objects
        self.parsed_files = set()
        self.logger = logging.getLogger(__name__)

    async def parse_files(self, files: List[BytesIO]):
        """
        Parse a list of files asynchronously.
        """
        tasks = []
        for file in files:
            if file.name in self.parsed_files:
                self.logger.info(f"File already parsed: {file.name}")
                continue

            if file.name.endswith(".pdf"):
                tasks.append(self.parse_pdf(file))
            elif file.name.endswith((".png", ".jpg", ".jpeg")):
                tasks.append(self.parse_image(file))
            elif file.name.endswith((".xlsx", ".csv")):
                tasks.append(self.parse_excel_csv(file))
            elif file.name.lower().endswith(tuple(self.supported_extensions)):
                tasks.append(self.parse_other(file))
            else:
                self.logger.warning(f"Unsupported file type: {file.name}")

            self.parsed_files.add(file.name)

        if tasks:
            await asyncio.gather(*tasks)

    async def parse_pdf(self, file):
        """
        Parse a PDF file asynchronously.
        """
        try:
            url = self.config.parse_api_url
            if not url:
                raise ValueError("PARSE_API_URL not set in configuration.")

            data = aiohttp.FormData()
            data.add_field(
                'file',
                file,
                filename=file.name,
                content_type='application/pdf'
            )

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(url, data=data, timeout=timeout) as response:
                    response_text = await response.text()
                    response_json = json.loads(response_text)

            extracted_texts = response_json.get("extracted_text", {}).get("0", [])
            # After extracting text chunks, create Document objects
            for i, chunk in enumerate(extracted_texts):
                doc = Document(
                    text=chunk,
                    metadata={
                        "filename": file.name,
                        "id": f"{file.name}_{i}",
                        "date_uploaded": datetime.utcnow().timestamp(),
                    }
                )
                self.parsed_documents.append(doc)  # Store Document object

            self.logger.info(f"Parsed PDF file: {file.name}")

        except Exception as e:
            self.logger.error(f"Error parsing PDF {file.name}: {e}")

    async def parse_image(self, file):
        """
        Parse an image file asynchronously.
        """
        try:
            url = self.config.parse_api_url
            if not url:
                raise ValueError("PARSE_API_URL not set in configuration.")

            data = aiohttp.FormData()
            data.add_field(
                'file',
                file,
                filename=file.name,
                content_type='application/octet-stream'
            )

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=300)
                async with session.post(url, data=data, timeout=timeout) as response:
                    response_text = await response.text()
                    response_json = json.loads(response_text)

            extracted_text = response_json.get("extracted_text", {}).get("0", [""])[0]
            doc_id = file.name
            doc = Document(
                text=extracted_text,
                metadata={
                    "filename": file.name,
                    "id": doc_id,
                    "date_uploaded": datetime.utcnow().timestamp(),
                }
            )
            self.parsed_documents.append(doc)  # Store Document object

            self.logger.info(f"Parsed image file: {file.name}")

        except Exception as e:
            self.logger.error(f"Error parsing image {file.name}: {e}")

    async def parse_excel_csv(self, file):
        """
        Parse Excel or CSV files asynchronously.
        """
        self.logger.info(f"Started parsing Excel/CSV file: {file.name}")
        try:
            if file.name.endswith(".xlsx"):
                df = await asyncio.to_thread(pd.read_excel, file)
            elif file.name.endswith(".csv"):
                file.seek(0)  # Ensure the file pointer is at the beginning
                df = await asyncio.to_thread(pd.read_csv, file)
            else:
                raise ValueError("Unsupported file format for Excel/CSV parser.")

            # Process DataFrame
            for index, row in df.iterrows():
                doc = Document(
                    text=row.to_json(),
                    metadata={
                        "filename": file.name,
                        "id": f"{file.name}_{index}",
                        "date_uploaded": datetime.utcnow().timestamp(),
                    }
                )
                self.parsed_documents.append(doc)

            self.logger.info(f"Finished parsing Excel/CSV file: {file.name}")

        except Exception as e:
            self.logger.error(f"Error parsing Excel/CSV {file.name}: {e}")


    async def parse_other(self, file):
        """
        Parse other supported file types asynchronously.
        """
        try:
            url = self.config.parse_api_url
            if not url:
                raise ValueError("PARSE_API_URL not set in configuration.")

            data = aiohttp.FormData()
            data.add_field(
                'file',
                file,
                filename=file.name,
                content_type='application/octet-stream'
            )

            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=120)
                async with session.post(url, data=data, timeout=timeout) as response:
                    response_text = await response.text()
                    response_json = json.loads(response_text)

            extracted_texts = response_json.get("extracted_text", {}).get("0", [])
            # After extracting text chunks, create Document objects
            for i, chunk in enumerate(extracted_texts):
                doc = Document(
                    text=chunk,
                    metadata={
                        "filename": file.name,
                        "id": f"{file.name}_{i}",
                        "date_uploaded": datetime.utcnow().timestamp(),
                    }
                )
                self.parsed_documents.append(doc)  # Store Document object

            self.logger.info(f"Parsed file: {file.name}")

        except Exception as e:
            self.logger.error(f"Error parsing file {file.name}: {e}")

    def get_parsed_documents(self):
        """
        Return the list of parsed Document objects.
        """
        return self.parsed_documents

    def generate_metadata(self):
        """
        Generate metadata for parsed documents.
        """
        if not self.parsed_documents:
            self.logger.error("No documents to generate metadata for. Please parse documents first.")
            return

        for doc_id, doc_data in self.parsed_documents.items():
            # Generate metadata such as title and summary (placeholder implementation)
            self.parsed_documents[doc_id].update({
                "title": f"Title for {doc_data['filename']}",
                "summary": f"Summary for document {doc_id}"
            })
            self.logger.info(f"Generated metadata for document: {doc_id}")

    def extract_user_needs(self, user_question: str) -> str:
        """
        Uses OpenAI GPT model to rephrase and extract key topics from the user's question.

        Args:
            user_question (str): The user's question.

        Returns:
            str: Rephrased question with key topics extracted.
        """
        try:
            if not self.config.openai_api_key:
                logging.error("OPENAI_API_KEY not set.")
                return ""

            # Use OpenAI's API to generate response
            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Rephrase the User Input. Extract key topics."},
                    {"role": "user", "content": f"User Input: {user_question}"}
                ],
                seed=42,
            )

            user_needs = response.choices[0].message.content.strip()
            return user_needs
        except Exception as e:
            logging.error(f"Error extracting user needs: {str(e)}")
            return ""

    def list_files(self, top_n=10, all_files=False):
        """
        List files with their metadata.

        Args:
            top_n (int): Number of recent files to return. Default is 10.
            all_files (bool): If True, return all files.

        Returns:
            List[Dict]: A list of dictionaries containing file metadata.
        """
        files = [
            {
                "filename": doc["filename"],
                "id": doc["id"],
                "date_uploaded": doc.get("date_uploaded", ""),
                "title": doc.get("title", ""),
                "summary": doc.get("summary", "")
            }
            for doc in self.parsed_documents.values()
        ]

        # Sort files by date_uploaded in descending order
        files.sort(key=lambda x: x.get("date_uploaded", ""), reverse=True)

        if all_files:
            return files
        else:
            return files[:top_n]