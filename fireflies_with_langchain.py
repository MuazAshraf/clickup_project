from flask import Flask, render_template, request, jsonify
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain
import openai, os, datetime
# Pinecone imports
import pinecone
from langchain.vectorstores import Pinecone
# OpenAI imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
# Chain imports
from langchain.chains.router import MultiRetrievalQAChain
# Agent imports
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
# Memory imports
from langchain.memory.buffer import ConversationBufferMemory
#clickup imports
from langchain.utilities.clickup import ClickupAPIWrapper
from langchain.agents.agent_toolkits.clickup.toolkit import ClickupToolkit


from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
oauth_client_id = os.getenv("oauth_client_id")
oauth_client_secret = os.getenv("oauth_client_secret")
redirect_uri = "mojosolo.com"
# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
	environment=PINECONE_API_ENV     
)
index_name = "mojosolo-main"

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()

# Declare agent as a global variable
agent = None
app = Flask(__name__)

API_ENDPOINT = "https://api.fireflies.ai/graphql"
API_KEY = "9aa5fb68-7963-448d-a257-2539cc5863fb"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
# Define your GraphQL query
query = """
{
  transcripts {
    id
    date
    sentences{
    text
    }
  }
}
"""

# Structure your request data
data = {
    "query": query
}

response = requests.post(API_ENDPOINT, headers=headers, json=data)

if response.status_code == 200:
    # Successfully received data
    print(response.json())
else:
    # Handle the error
    print(f"Failed to fetch data from Fireflies. Status Code: {response.status_code}")


def get_latest_transcript_from_fireflies():
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    
    if response.status_code == 200:
        # Extract the transcript data from the response 
        latest_transcript = response.json().get('data').get('transcripts')[0]
        
        # Concatenate the sentences to form the full transcript text
        sentences = latest_transcript.get('sentences', [])
        full_transcript_text = " ".join(sentence['text'] for sentence in sentences)
        
        # Add the concatenated text to the latest_transcript dictionary
        latest_transcript['text'] = full_transcript_text
        return latest_transcript   
    else:
        print("Failed to fetch data from Fireflies. Status Code:", response.status_code)
        return None

transcript_summary_template = """
You are an AI assistant. Given the transcript text below, describe the detailed summary of the transcript:

Transcript:
{input}
"""
summary_prompt_template = PromptTemplate(input_variables=["input"], template=transcript_summary_template)
summary_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-3.5-turbo-16k"), prompt=summary_prompt_template)

project_plan_template = """
You are an AI assistant. Given the transcript text below, extract the project plan:

Transcript:
{input}
"""
project_plan_prompt_template = PromptTemplate(input_variables=["input"], template=project_plan_template)
project_plan_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-3.5-turbo-16k"), prompt=project_plan_prompt_template)

# Update the PromptTemplate and LLMChain setup for Project Deadline
project_deadline_template = """
You are an AI assistant. Given the transcript text below, extract the project deadline:

Transcript:
{input}
"""
project_deadline_prompt_template = PromptTemplate(input_variables=["input"], template=project_deadline_template)
project_deadline_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-3.5-turbo-16k"), prompt=project_deadline_prompt_template)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_transcript', methods=['POST'])
def get_transcript():
    latest_transcript = get_latest_transcript_from_fireflies()
    if latest_transcript:
        overall_chain = SimpleSequentialChain(chains=[summary_chain,project_plan_chain, project_deadline_chain], verbose=True)
        overall_chain.run({"input": latest_transcript['text']})


        
        # Assume each chain result is a single string containing the extracted info
        summary_result = summary_chain.run({"input": latest_transcript['text']})
        project_plan_result = project_plan_chain.run({"input": latest_transcript['text']})
        project_deadline_result = project_deadline_chain.run({"input": latest_transcript['text']})

        return jsonify({
            'transcript': latest_transcript['text'],
            'summary': summary_result,
            'project_plan': project_plan_result,
            'project_deadline': project_deadline_result,
        })
    else:
        return jsonify({'error': 'Failed to fetch transcript'}), 500

def upsertToPinecone(mem, namespace):
    Pinecone.from_texts(texts=[mem], index_name=index_name, embedding=embeddings, namespace=namespace)
    return "Saved " + mem + " to client database"
    

@app.route('/save_to_pinecone', methods=['POST'])
def save_to_pinecone():
    namespace = request.form['namespace']
    transcript_data = request.form['transcript']
    response = upsertToPinecone(transcript_data, namespace)
    success = "Saved" in response
    return jsonify({"success": success})

@app.route('/clickup_agent')
def clickup_agent():
    return render_template('clickup_agent.html')
# Here's where you place the print_and_run function
def print_and_run(command):
    if agent is None:
        print("Error: Agent is not initialized.")
        return

    print('\033[94m$ COMMAND\033[0m')
    print(command)
    print('\n\033[94m$ AGENT\033[0m')
    response = agent.run(command)
    print(response)
    print(''.join(['-']*80))
    return response
@app.route('/initialize_agent', methods=['POST'])
def initialize_agent_route():
    global agent
    data = request.json
    oauth_client_id = data.get('oauth_client_id')
    oauth_client_secret = data.get('oauth_client_secret')
    code = data.get('code')
    access_token = ClickupAPIWrapper.get_access_token(oauth_client_id, oauth_client_secret, code)
    clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
    toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(toolkit.get_tools(), llm=llm, memory=memory, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return jsonify(status='success', message='Agent Initialized successfully')

@app.route('/run_command', methods=['POST'])
def run_command_route():
    if not agent:
        return jsonify(status='error', message='Agent is not initialized.')
    
    data = request.json
    command = data.get('command')
    print("command",command)
    response = agent.run(command)
    return jsonify(response=response)

    

if __name__ == "__main__":
    time_str = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    print_and_run(f"Create a task called 'Test Task - {time_str}' with description 'This is a Test'")
    app.run(debug=True)
