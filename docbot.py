import re

from dotenv import load_dotenv
import requests

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import DeepLake

from prompts import (
    chat_prompt,
    combine_prompt_template,
    link_filter_message,
    question_prompt_template,
)
from utils import html_as_markdown

load_dotenv()


def build_qa(vector_store, llm):
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )
    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT,
        },
    )

    return qa


def fetch_resources(question, messages, vector_store):
    messages.extend(chat_prompt.format_prompt(question=question).to_messages())
    response = chat(messages)
    messages.append(AIMessage(content=response.content))

    print("Fetching references from the internet...")
    response = requests.get(f"https://www.google.com/search?q={response.content}")
    as_markdown = html_as_markdown(response.content)

    messages.extend(link_filter_message.format_messages(markdown=as_markdown))
    response = chat(messages)

    # Okay, now we have URLs to resources. Let's fetch them all and store them in our
    # document store
    urls = re.findall("https?://[^\s]+", response.content)
    for url in urls:
        resp = requests.get(url)
        if resp.status_code == 200:
            mdown = html_as_markdown(resp.content)
        else:
            continue

        print(f"Saving reference documentation at {url} in Deep Lake store.\n\n")
        docs = markdown_splitter.create_documents([mdown])
        vector_store.add_documents(docs, verbose=False)


llm = OpenAI(temperature=0.1)
chat = ChatOpenAI(temperature=0.1)
embeddings = OpenAIEmbeddings()
db = DeepLake(dataset_path="./datalake/", embedding_function=embeddings, verbose=False)
qa_chain = build_qa(db, llm)
markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=20)

if __name__ == "__main__":
    messages = []
    while True:
        question = input(
            "\nI am your friendly documentation bot. Please ask a question, I'll do my best to answer.\n\n> "
        )
        fetch_question = input(
            "Should I fetch resources before answering? [Yes/yes/Y/y, No/N/no/n]\n\n> "
        )
        if fetch_question.lower().startswith("y"):
            fetch_resources(question, messages, db)

        print("One second, thinking...\n")

        # This queries the vector store for relevant documents, then passes them to the
        # LLM along with the user's question to answer.
        answer = qa_chain.run(question)

        print(answer + "\n\n")
        messages.append(AIMessage(content=answer))

        fetch_question = input(
            "Should I fetch resources and try again? [Yes/yes/Y/y, No/N/no/n]\n\n> "
        )
        if fetch_question.lower().startswith("y"):
            messages.append(
                HumanMessage(content=
                    "That's not quite what I want. Please search for more resources."
                )
            )
            fetch_resources(question, messages, db)
            answer = qa_chain.run(question)
            print(answer + "\n\n")
