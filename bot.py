import re

from dotenv import load_dotenv
import requests

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
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
    def __init__(
        self,
        streamlit,
        openai_key=None,
        dataset_path=None,
        temperature=0.2,
        max_message_tokens=3000,
        similiarity_score_threshold=0.6,
        debug=False,
    ):
        self.streamlit = streamlit
        self.max_message_tokens = max_message_tokens
        self.similarity_score_threshold = similiarity_score_threshold
        self.debug = debug

        self.chat = ChatOpenAI(openai_api_key=openai_key, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.db = DeepLake(
            dataset_path="./datalake/" if dataset_path is None else dataset_path,
            embedding_function=self.embeddings,
            verbose=False,
        )
        self.qa_chain = build_qa(self.db, self.chat)
        self.markdown_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=20)
        self.messages = []

        if self.debug:
            self.db.ds.summary()

    def check_vectordb(self, question):
        try:
            docs = self.db.similarity_search(question, return_score=True)
        except Exception as e:
            print(e)
            return 0
        docs = [d for d in docs if d[1] < self.similarity_score_threshold]
        return len(docs)

    def _prune_messages(self):
        while (
            self.chat.get_num_tokens_from_messages(self.messages)
            > self.max_message_tokens
        ):
            self.messages.pop(0)

    def _run_model(self):
        self._prune_messages()
        response = self.chat(self.messages)
        self.messages.append(AIMessage(content=response.content))
        return response

    def generate_search_phrase(self, question):
        self.messages.extend(search_phrase_message.format_messages(question=question))
        response = self._run_model()
        search_phrase = response.content.replace('"', "")
        self.streamlit.write(search_phrase) if self.debug else None
        return search_phrase

    def fetch_resources(self, search_phrase):
        response = requests.get(
            f"https://www.google.com/search?q={'+'.join(search_phrase.split(' '))}"
        )
        as_markdown = html_as_markdown(response.content)

        self.messages.extend(link_filter_message.format_messages(markdown=as_markdown))
        response = self._run_model()

        # Okay, now we have URLs to resources. Let's fetch them all and store them in
        # the document store
        urls = re.findall("https?://[^\s]+", response.content)
        self.streamlit.write(urls) if self.debug else None
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

            docs = self.markdown_splitter.create_documents(
                [mdown], metadatas=[{"source": url}]
            )
            self.streamlit.write(f"Adding resource from {url}") if self.debug else None
            self.db.add_documents(docs, progressbar=False)
        return True

    def qa_run(self, question: str):
        self.add_human_message(
            f"I'm going to give you a bunch of documents so you can answer my question. Again, here's the question: {question}"
        )
        self._prune_messages()
        answer = self.qa_chain.run(question)
        self.messages.append(AIMessage(content=answer))
        return answer

    def add_human_message(self, text):
        self.messages.append(HumanMessage(content=text))
