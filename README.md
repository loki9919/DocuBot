## Introduction
------------
The TalkToPDF Chat App is a Python application that allows you to chat with your own PDF documents. One can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works
------------
The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their textual content.

2. Text Chunking: The extracted text is divided into smaller, manageable chunks for efficient processing..

3. Language Model: The application uses a language model to create vector representations (embeddings) of the text chunks..

4. Similarity Matching: When a question is asked, the app compares it with the text chunks to find the most semantically similar ones.

5. Response Generation: The selected chunks are fed into the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
To install the ALXGPT Chat App, please follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/loki9919/TalkToPDF.git
   cd TalkToPDF

2. Install the required dependencies by running the following command:
   ```bash
   pip install -r requirements.txt
   

3. Obtain an API key from OpenAI and also for Llamaparser add it to the .env file in the project directory.
commandline
    ```bash
    OPENAI_API_KEY=your_secrit_api_key
    LLAMA_CLOUD_API_KEY=yout_secrit_api_key

## Usage
-----
To use the ALXGPT Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and add your keys to .env file.

2. Run the app.py file using the Streamlit CLI. Execute the following command:
   ```bash
   streamlit run app.py
   

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## License
-------
The TalkToPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).