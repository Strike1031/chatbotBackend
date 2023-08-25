#openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

#pinecone
import pinecone
from langchain.vectorstores import Pinecone

#tools
from langchain.tools import Tool

#agente
from langchain.agents import initialize_agent

# retrieval
from langchain.chains import RetrievalQA

#news tool
from langchain.utilities import GoogleSerperAPIWrapper

#retriever
from langchain.chains import RetrievalQA

#memory_and_prompt
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#os
import os
from dotenv import load_dotenv

load_dotenv()

# setting keys and environments

# os.environ['OPENAI_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = "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"
os.environ["SERPER_API_KEY"] = "f3b784c2667848e4b562af080d7a6ac397984cbd"
pinecone_api_key = "50f2bb6e-12f5-4b29-9c5f-546cf631a64b"
pinecone_environment = "us-west1-gcp-free"

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.0,
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
)

# setting pinecone

embeddings = OpenAIEmbeddings()
pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = 'vegaindex'
vectorstore = Pinecone.from_existing_index(index_name, embeddings)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

system_template = """
You are Groot, the dedicated assistant of the Vega Crypto company. As the
intellectual powerhouse behind the scenes, you empower employees by swiftly
addressing their inquiries and guiding them through various tasks. Your
extensive knowledge base allows you to respond adeptly to a wide array of topics
pertinent to both the company and general subjects.

Groot, your personality exudes charisma and politeness, making interactions with
users a delightful experience. Your commitment to linguistic accuracy ensures
that you always respond in the language they communicate, providing efficient
and effective solutions. If ever you encounter a question you can't answer, you
openly admit your limitation.

In the event of any inaccuracies, you sincerely apologize and readily accept
correction from users. If a user identifies incorrect information, your
charismatic nature shines as you gracefully apologize and correct the mistake.

Remember, Groot, your primary goal is to serve as a valuable asset to the
Vega Crypto company, enriching interactions and empowering users with your
unwavering knowledge and pleasant demeanor."""

system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_prompt = HumanMessagePromptTemplate.from_template("{input}")

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# news tools

import pprint
from langchain.document_loaders import WebBaseLoader
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.text_splitter import TokenTextSplitter

prompt_template = """Write a concise summary of the text below, write the summary in the language of the text, and write as if you were a professional reporter:
"{text}"
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="text"
)

def newsSearch(q):
  search  = GoogleSerperAPIWrapper(type="news", tbs="qdr:w", gl='br', hl='pt-br')
  results = search.results(q)['news']


  links = [link['link'] for link in results]
  print(links)
  headlines = [headline['title'] for headline in results]
  news = ''

  for link, headline in zip(links, headlines):
    loader = WebBaseLoader(link)
    docs = loader.load_and_split()

    news += 'title: ' + headline + '\n' + stuff_chain.run(docs) + '\nfont: ' + link + '\n-----------------------------------------------------------------------------\n'
  return news

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description='Utilize the Knowledge Base tool to fetch answers directly from documents. All queries should looking for information using the Document search tool first.'
    ),
    Tool(
        name = "News Tool",
        func=lambda q: str(newsSearch(q)),
        description='Use the News Tool to get current headlines from the internet. All queries that ask for news and related things should use this tool.',
        return_direct=True,

    ),
]

# agent = initialize_agent(
#     agent='chat-conversational-react-description',
#     tools=tools,
#     llm=llm,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method='generate',
#     memory=conversational_memory,
# )

user_conversations = {}

def get_response(user_id, text):
    if user_id not in user_conversations:
        # Initialize conversation history for new user
        user_conversations[user_id] =  initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory,
        )

    user_agent = user_conversations[user_id]

    try:
        response = user_agent(chat_prompt.format_prompt(input=text).to_string())['output']

        return response
    except Exception as e:
        error_message = f"I'm sorry, I can't respond to your message due to an error in my system: {e}"
        return error_message