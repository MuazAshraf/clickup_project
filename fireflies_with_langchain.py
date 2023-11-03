from flask import Flask, render_template, request, jsonify, abort
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
# After you import Flask
from flask_cors import CORS

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv('.env', override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
redirect_uri = "mojosolo.com"
# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
	environment=PINECONE_API_ENV     
)
#index_name = "mojosolo-main"
index_name = "testing"

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


app = Flask(__name__)

CORS(app)
API_ENDPOINT = "https://api.fireflies.ai/graphql"
API_KEY = "9aa5fb68-7963-448d-a257-2539cc5863fb"
fireflies_headers = {
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

response = requests.post(API_ENDPOINT, headers=fireflies_headers, json=data)

# if response.status_code == 200:
#     # Successfully received data
#     print(response.json())
# else:
#     # Handle the error
#     print(f"Failed to fetch data from Fireflies. Status Code: {response.status_code}")


def get_latest_transcript_from_fireflies():
    response = requests.post(API_ENDPOINT, headers=fireflies_headers, json=data)
    
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
summary_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-4"), prompt=summary_prompt_template)

project_plan_template = """
You are an AI assistant. Given the transcript text below, extract the project plan:

Transcript:
{input}
"""
project_plan_prompt_template = PromptTemplate(input_variables=["input"], template=project_plan_template)
project_plan_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-4"), prompt=project_plan_prompt_template)

# Update the PromptTemplate and LLMChain setup for Project Deadline
project_deadline_template = """
You are an AI assistant. Given the transcript text below, extract the project deadline:

Transcript:
{input}
"""
project_deadline_prompt_template = PromptTemplate(input_variables=["input"], template=project_deadline_template)
project_deadline_chain = LLMChain(llm=ChatOpenAI(temperature=.7,model="gpt-4"), prompt=project_deadline_prompt_template)

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


# Declare agent as a global variable
agent = None
@app.route('/initialize_agent', methods=['POST'])
def initialize_agent_route():
    global agent
    # oauth_client_id = "ZSE6HZSO5NH9H308RC0K28E69TXNFZU2"
    # oauth_client_secret = "H77LL3CNCGXU6OD2JU3J5H0YZJQ28FM1987NPQS3WJPJTNZ2VUTN1C96WJFEKRIW"
    # code = "OP8LX1B11BKIUJAH9YSAG89C47BDBM91"
    # print("Client ID:", oauth_client_id)
    # print("Client Secret:", oauth_client_secret)
    # print("Code:", code)
    #access_token = ClickupAPIWrapper.get_access_token(oauth_client_id, oauth_client_secret, code)
    access_token = "44183335_507a749059557a8f5a99973e7cb8f6c85fe38a6f627cf651b934ac86dcfa98ea"
    clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
    toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)
    print(f'Found team_id: {clickup_api_wrapper.team_id}.\nMost request require the team id, so we store it for you in the toolkit, we assume the first team in your list is the one you want. \nNote: If you know this is the wrong ID, you can pass it at initialization.')
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    agent = initialize_agent(
        toolkit.get_tools(), llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return jsonify(status='success', message='Agent Initialized successfully')

def print_and_run(command):
    if agent is None:
        print("Agent is not initialized.")
        return

    response = agent.run(command)
    return str(response)  # Convert the response to a string


def create_structure_with_agent(folderName, listName, taskName, description, assigneeName):
    if agent is None:
        return "Agent is not initialized."

    # Create a folder
    folder_response = print_and_run(f"Create a folder named '{folderName}'")

    # Create a list inside the folder
    list_response = print_and_run(f"No matter what but Inside that'{folderName}' folder, create a list name '{listName}'")

    # Create a task inside the list
    task_response = print_and_run(
    f"Inside the '{listName}' list, create a task called '{taskName}' "f"with description '{description}', and assign it to {assigneeName}"
)  # Use assigneeName

    return jsonify(folder_response=folder_response, list_response=list_response, task_response=task_response)
# Replace 'your_secret_token' with your actual secret token
SECRET_TOKEN = 'MojoDeadWalking'
@app.route('/create_structure_with_agent', methods=['POST'])
def create_structure_route():
    # Directly access the Authorization header
    incoming_token = request.headers.get('Authorization')

    # Print the incoming token
    print("Incoming Token:", incoming_token)

    # Check if the incoming token is correct
    if incoming_token != SECRET_TOKEN:
        return jsonify(error="Unauthorized access", message="Invalid token"), 401

    data = request.json
    folderName = data.get('folderName')
    listName = data.get('listName')
    taskName = data.get('taskName')
    description = data.get('description')
    assigneeName = data.get('assigneeName')  # Use assigneeName

    response = create_structure_with_agent(folderName, listName, taskName, description, assigneeName)  # Use assigneeName

    # Since response is a Flask Response object, you can return it directly
    return response




roles = {
    "David Matenaer": ["Agency Principal", "Creative Director", "Innovation/Workflow Expert"],
    "Kira Diner": ["Senior Account Executive", "COO", "Producer/Strategy"],
    "Chris Benjamin": ["HR", "Junior Account Executive"],
    "Erin Abbott":["Project Manager", "Junior Account Executive"],
    "Eric Lindholm":["SCRUM Master"],
    "Zain Zulifqar":["Senior Developer back-end"],
    "Saif Sohail":["Junior Front-End Developer"],
    "Muaz Ashraf":["Junior AI/NLP/ML Developer"],
    "Nick Michael":["Videographer, Video Editor"],
    "Chandrick":["Senior Motion Graphics Designer/Editor"],
    "Henry Hoegland":["AI/NLP/ML Laravel Developer Intern/Strategy"],
    "McKinzie Plotts":["Senior Graphic Designer Part-time]"],
    "Ian Matenaer":["Front-End Developer Intern"],
    "Maggie Heilmann":["Office/Account Manager"]
}
def get_roles_for_user(name):
    return roles.get(name, [])

@app.route('/get_roles/<name>', methods=['GET'])
def get_roles(name):
    user_roles = get_roles_for_user(name)
    if user_roles:
        return jsonify(status='success', roles=user_roles)
    else:
        return jsonify(status='error', message=f'No roles found for {name}')

if __name__ == "__main__":
    app.run(port = 5001, debug=True)
