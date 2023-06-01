import os

import streamlit as st

from bot import Bot

if "do_search" not in st.session_state:
    st.session_state["do_search"] = False


@st.cache_resource
def load_bot(
    openai_key: str | None, dataset_path: str | None = None, temperature: float = 0.2
):
    return Bot(
        st,
        openai_key=openai_key,
        temperature=temperature,
        debug=os.environ.get("DEBUG", False),
    )


### Sidebar ###
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
dataset_path = st.sidebar.text_input("Dataset path")
activeloop_token = st.sidebar.text_input("Activeloop Token", type="password")
st.sidebar.divider()
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)

if activeloop_token:
    os.environ["ACTIVELOOP_TOKEN"] = activeloop_token

### Main Content ###
bot = load_bot(openai_key, dataset_path, temperature)

st.title("All Your Questions Answered")


st.write(
    "Hello, I'm your friendly help bot. Ask me a question! If I need more "
    "knowledge to give you a good answer, I'll search the internet for "
    "information."
)

question = st.text_input("Your question...")

if question:
    n_docs = bot.check_vectordb(question)

    if n_docs < 4 or st.session_state.do_search:
        if st.session_state.do_search:
            bot.add_human_message(
                "That's not quite what I was looking for. Refine the search phrase to find more specific resources you can use to answer my question."
            )
        with st.spinner("Searching the internet for more resources..."):
            search_phrase = bot.generate_search_phrase(question)
            bot.fetch_resources(search_phrase)
            st.session_state.do_search = False

    with st.spinner("Searching the knowledge base, one second..."):
        answer = bot.qa_run(question)

    st.divider()
    st.markdown(answer)


def force_search():
    st.session_state.do_search = True


search_button = st.button("Search for more resources", on_click=force_search)
