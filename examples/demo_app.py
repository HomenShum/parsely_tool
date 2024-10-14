# examples/demo_app.py

import streamlit as st
import asyncio
from parsely_tool import Parser, Storage, QueryEngine, Utils, Config
from io import BytesIO
import logging

def main():
    st.title("Parsely Demo App")

    # Setup configuration
    config = Config(username='username', verbose=True)
    Utils.setup_logging(config.verbose)
    logger = logging.getLogger(__name__)

    # Initialize components
    parser = Parser(config)
    storage = Storage(config)
    query_engine = QueryEngine(config)

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files",
        type=parser.supported_extensions,
        accept_multiple_files=True
    )

    if st.button("Process Files"):
        if uploaded_files:
            file_objects = []
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.read()
                file_obj = BytesIO(file_content)
                file_obj.name = uploaded_file.name
                file_objects.append(file_obj)

            # Parse and store files
            asyncio.run(parser.parse_files(file_objects))
            storage.store_documents(parser.get_parsed_documents())
            st.success("Files processed successfully!")
        else:
            st.warning("No files uploaded.")

    # Query input
    query_text = st.text_input("Enter your query:")
    if st.button("Search"):
        if query_text:
            results = query_engine.query(query_text)
            st.write("Query Results:")
            for result in results:
                st.write(result)
        else:
            st.warning("Please enter a query.")

if __name__ == '__main__':
    main()
