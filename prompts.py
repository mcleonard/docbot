from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

search_phrase_message = HumanMessagePromptTemplate.from_template(
    """
A user is attempting to answer a question using technical documentation
and other resources found on the internet. You will return a phrase that
can be used in a search engine to find references to answer the question.

Question: {question}
The search phrase:
"""
)

link_filter_message = HumanMessagePromptTemplate.from_template(
    """
From the following Markdown content, please return the five most relevant links for
answering the user's question. Leave out any YouTube videos.

Markdown: {markdown}

Return only the URLs without bullets, one on each line.
URLs:
"""
)

chat_prompt = ChatPromptTemplate.from_messages([search_phrase_message])

question_prompt_template = """You are a help bot for an open source software community.
Use this section of the documentation to answer a user's question, if the text is 
related to the question:
{context}
Question: {question}
Please include any Python code that is relevant to the answer.
Relevant text, if any:"""


combine_prompt_template = """Given the following summaries from the framework 
documentation, collect them into one combined answer for the user's question. Include 
Python code if relevant. If you don't have a good answer, just say that you don't know.
Summaries:
{summaries}
Question: {question}
Combined answer, if any:
"""