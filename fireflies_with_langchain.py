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
redirect_uri = "mojosolo.com"
# Initialize pinecone and set index
pinecone.init(
    api_key= PINECONE_API_KEY,      
	environment=PINECONE_API_ENV     
)
#index_name = "mojosolo-main"
index_name = "testing"

# Initialize embeddings and AI
embeddings = OpenAIEmbeddings()


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


# Declare agent as a global variable
agent = None
@app.route('/initialize_agent', methods=['POST'])
def initialize_agent_route():
    global agent
    oauth_client_id = "57NNFTN3Q19ECB4GHTQ2E3ZC6W7KO8G2"
    oauth_client_secret = "9CBIEKCMB2NSVXEQZ0AIPZK32V5HNEOUZA4PFFHQ7BVUXNHAVZSN1SV5OD5K8TY8"
    code = "G4AKQBCSGKED6QV3UFMAY43B4SOBGRWK"
    print("Client ID:", oauth_client_id)
    print("Client Secret:", oauth_client_secret)
    print("Code:", code)
    #access_token = ClickupAPIWrapper.get_access_token(oauth_client_id, oauth_client_secret, code)
    access_token = "72038349_e1efa1989a7901c8abcc84bd2dbc0ad6c9ef422b2278bbfd034cca1717e92255"
    clickup_api_wrapper = ClickupAPIWrapper(access_token=access_token)
    toolkit = ClickupToolkit.from_clickup_api_wrapper(clickup_api_wrapper)
    print(f'Found team_id: {clickup_api_wrapper.team_id}.\nMost request require the team id, so we store it for you in the toolkit, we assume the first team in your list is the one you want. \nNote: If you know this is the wrong ID, you can pass it at initialization.')
    llm = ChatOpenAI(temperature=0, openai_api_key="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")
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(
        toolkit.get_tools(), llm=llm, memory = memory, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    return jsonify(status='success', message='Agent Initialized successfully')


# helper function for demo
def print_and_run(command):
    if agent is None:
        print("Agent is not initialized.")
        return
    print('\033[94m$ COMMAND\033[0m')
    print(command)
    print('\n\033[94m$ AGENT\033[0m')
    response = agent.run(command)
    print(''.join(['-']*80))
    return response

print_and_run("Get all the teams that the user is authorized to access")
print_and_run("Get all the spaces available to the team")
print_and_run("Get all the folders for the team")


@app.route('/create_folder', methods=['POST'])
def create_folder_route():
    data = request.json
    folder_name = data.get('folder_name')
    
    if agent is None:
        return jsonify(status='error', message='Agent is not initialized.')
    
    command = f"Create a folder called '{folder_name}'"
    response = print_and_run(command)
    print("Folder creation response:", response)
    return jsonify(response=response)


@app.route('/create_list', methods=['POST'])
def create_list_route():
    if agent is None:
        return jsonify(status='error', message='Agent is not initialized.')

    data = request.json
    listName = data.get('listName')
    priority = data.get('priority')

    # Construct the command dynamically
    command = f"Create a list called '{listName}' with priority {priority}"

    response = print_and_run(command)
    print("List creation response:", response)
    return jsonify(response=response)
@app.route('/create_task', methods=['POST'])
def create_task_route():
    if agent is None:
        return jsonify(status='error', message='Agent is not initialized.')

    data = request.json
    taskName = data.get('taskName')
    description = data.get('description')
    assigneeName = data.get('assigneeName')

    # Construct the command dynamically
    command = f"Figure out what user ID {assigneeName} is, create a task called '{taskName}', description '{description}', then assign it to {assigneeName}"

    response = print_and_run(command)
    print("Task creation response:", response)
    return jsonify(response=response)
@app.route('/execute_command', methods=['POST'])
def execute_command_route():
    if agent is None:
        return jsonify(status='error', message='Agent is not initialized.')

    data = request.json
    command = data.get('command')
    response = print_and_run(command)
    print("Command creation response:", response)
    return jsonify(response=response)

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
    app.run(host='0.0.0.0', port=5000, debug=True)

