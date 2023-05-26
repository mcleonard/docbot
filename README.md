# DocBot

This a Streamlit app that leverages cool new AI things to answer questions. It searches the internet for resources and collects information from multiple sources into one coherent answer. It uses a [Deep Lake vector store](https://docs.activeloop.ai) as a knowledge base and LangChain to interact with GPT-3 for searching the internet and the knowledge base to answer questions.

## DocBot Design

This bot consists of two systems. One system searches the internet for resources such as documentation, blog posts, and tutorials relevant to your question then stores them in a Deep Lake vector store. This vector store becomes a knowledge base for your work and life. This system uses `GPT-3.5-Turbo` to refine the searches and better understand which resources are relevant to you.

The other system queries the vector store for documents relevant to your question and uses GPT-3 to extract information from each of those documents and combines the information into one well-informed answer.

## Installation

Use [Poetry](https://python-poetry.org) to install the dependencies. Poetry installation [instructions here](https://python-poetry.org/docs/#installation).

Clone this repo:
```
git clone https://github.com/mcleonard/docbot.git
```

Then with Poetry installed:
```
cd docbot
poetry install
```

You'll need Python 3.10 or greater. You'll also need an [OpenAI API key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key). Set it as an enviroment variable, or define it in a `.env` file.

```
export OPENAI_API_KEY=<your key here>
```

or in a file called `.env` in the same directory as the `app.py` script:
```
OPENAI_API_KEY=<your_key_here>
```

## Usage
Run it with Poetry:
```
poetry run streamlit run app.py
```



