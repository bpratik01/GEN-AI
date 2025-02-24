from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langserve import add_routes

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
  model = 'gpt-4o-mini',
  temperature = 0,
  openai_api_key = OPENAI_API_KEY
)

# Define the prompt template

prompt = ChatPromptTemplate.from_messages([
  ('system', 'You are a helpful assistant.'),
  ('user', '{input}')
])

parser = StrOutputParser()

# Create the chain

chain = prompt | llm | parser

# Create the FastAPI app
app = FastAPI(
              title="Langserve",
              version="0.0.1",
              description="A simple API for Langchain"
            ) 

# adding chain route
add_routes(
  app, 
  chain, 
  path="/chain"
) 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)


 