import streamlit as st

from bot import Bot

st.title("All Your Questions Answered")

bot = Bot()

st.write(
    "Hello, I'm your friendly help bot. Ask me a question! If I need more "
    "knowledge to give you a good answer, I'll search the internet for more "
    "information."
)

question = st.text_input("Your question...")
do_search = False

if question:
    n_docs = bot.check_vectordb(question)

    if n_docs < 4 or do_search:
        if do_search:
            bot.add_human_message(
                "That's not quite what I was looking for. Please retrieve more resources you can use to answer my question."
            )
        with st.spinner("Searching the internet for more resources..."):
            search_phrase = bot.generate_search_phrase(question)
            bot.fetch_resources(search_phrase)

    with st.spinner("Searching the knowledge base, one second..."):
        answer = bot.qa_chain.run(question)

    st.divider()
    st.markdown(answer)

do_search = st.button("Search for more resources")
