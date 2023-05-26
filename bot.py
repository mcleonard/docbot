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
    search_phrase_message,
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


class Bot:
    def __init__(self):
        self.llm = OpenAI(temperature=0.2, max_tokens=2046)
        self.chat = ChatOpenAI(temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        self.db = DeepLake(
            dataset_path="./datalake/",
            embedding_function=self.embeddings,
            verbose=False,
        )
        self.qa_chain = build_qa(self.db, self.llm)
        self.markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.messages = []

    def check_vectordb(self, question):
        try:
            docs = self.db.similarity_search(question, return_score=True)
        except Exception as e:
            print(e)
            return 0
        print(docs)
        docs = [d for d in docs if d[1] < 0.6]
        return len(docs)

    def generate_search_phrase(self, question):
        self.messages.extend(search_phrase_message.format_messages(question=question))
        response = self.chat(self.messages)
        self.messages.append(AIMessage(content=response.content))

        search_phrase = response.content.replace('"', "")

        return search_phrase

    def fetch_resources(self, search_phrase):
        response = requests.get(
            f"https://www.google.com/search?q={'+'.join(search_phrase.split(' '))}"
        )
        as_markdown = html_as_markdown(response.content)

        self.messages.extend(link_filter_message.format_messages(markdown=as_markdown))
        response = self.chat(self.messages)

        # Okay, now we have URLs to resources. Let's fetch them all and store them in our
        # document store
        urls = re.findall("https?://[^\s]+", response.content)
        for url in urls:
            try:
                resp = requests.get(url)
            except requests.exceptions.ConnectionError:
                continue
            if resp.status_code == 200:
                mdown = html_as_markdown(resp.content)
            else:
                continue

            if mdown is None:
                continue

            print(f"Saving reference documentation at {url} in Deep Lake store.\n\n")
            docs = self.markdown_splitter.create_documents(
                [mdown], metadatas=[{"source": url}]
            )
            self.db.add_documents(docs, verbose=False)
        return True

    def add_human_message(self, text):
        self.messages.append(
            HumanMessage(
                content="That's not quite what I want. Please search for more resources."
            )
        )


if __name__ == "__main__":
    messages = []
    bot = Bot()
    while True:
        question = input(
            "\nI am your friendly documentation bot. Please ask a question, I'll do my best to answer.\n\n> "
        )
        fetch_question = input(
            "Should I fetch resources before answering? [Yes/yes/Y/y, No/N/no/n]\n\n> "
        )
        if fetch_question.lower().startswith("y"):
            bot.fetch_resources(question)

        print("One second, thinking...\n")

        # This queries the vector store for relevant documents, then passes them to the
        # LLM along with the user's question to answer.
        answer = bot.qa_chain.run(question)

        print(answer + "\n\n")
        bot.messages.append(AIMessage(content=answer))

        fetch_question = input(
            "Should I fetch resources and try again? [Yes/yes/Y/y, No/N/no/n]\n\n> "
        )
        if fetch_question.lower().startswith("y"):
            messages.append(
                HumanMessage(
                    content="That's not quite what I want. Please search for more resources."
                )
            )
            bot.fetch_resources(question, messages)
            answer = bot.qa_chain.run(question)
            print(answer + "\n\n")
