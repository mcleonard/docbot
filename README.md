# DocBot

This a bot that leverages cool new AI things to answer questions. It's mostly intended to use technical documentation, but it can honestly answer most things. It is able to search the internet for resources and collect information from multiple sources into one coherent answer. It's mostly a demonstration of using LangChain and Deep Lake, could be turned into something much nicer.

It works by storing resources fetched from the internet (Google searches) in a [Deep Lake vector store](https://docs.activeloop.ai), given a question. It queries the vector store using your question and sends relevant documents to GPT-3 to collect and summarize an answer.

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

or in a file called `.env` in the same directory as the `docbot.py` script:
```
OPENAI_API_KEY=<your_key_here>
```

## Usage
Run it with Poetry:
```
poetry run python docbot.py
```

then follow the prompts.



