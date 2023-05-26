from langchain.prompts.chat import HumanMessagePromptTemplate


search_phrase_message = HumanMessagePromptTemplate.from_template(
    """
A user is attempting to answer a question using documentation, blog posts,
tutorials, and other resources found on the internet. You will return a phrase that
can be used in a search engine to find references to answer the question.

Question: {question}
The search phrase:
"""
)

link_filter_message = HumanMessagePromptTemplate.from_template(
    """
From the following Markdown content, please return the most relevant links for
answering the user's question. Leave out any YouTube videos.

Markdown: {markdown}

Return only the URLs without bullets, one on each line.
URLs:
"""
)


question_prompt_template = """You are a bot that uses documentation, blog posts,
tutorials, and other resources to answer questions. Use this section of the documents
to answer the user's question, if the text is related to the question:
{context}
Question: {question}
Relevant text, if any:"""


combine_prompt_template = """Given the following summaries from Markdown documents, 
collect them into one combined answer for the user's question.  If you don't have a 
good answer, just say that you don't know.
Summaries:
{summaries}
Question: {question}
Combined answer, as Markdown. If there is code, format it correctly, and escape 
symbols like $ appropriately:
"""
