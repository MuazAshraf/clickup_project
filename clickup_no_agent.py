from flask import Flask, render_template, request, jsonify
import requests
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain
import os
# Pinecone imports
import pinecone
from langchain.vectorstores import Pinecone
# OpenAI imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

CLICKUP_TOKEN = '44183335_507a749059557a8f5a99973e7cb8f6c85fe38a6f627cf651b934ac86dcfa98ea'
BASE_URL = "https://api.clickup.com/api/v2/"
HEADERS = {
    "Authorization": CLICKUP_TOKEN,
    "Content-Type": "application/json"
}


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
    return render_template('clickup.html')  #noagent.html

def create_folder(space_id, name):
    url = f"{BASE_URL}space/{space_id}/folder"
    data = {
        "name": name
    }
    response = requests.post(url, headers=HEADERS, json=data)
    return response.json()

def create_list(folder_id, name):
    url = f"{BASE_URL}folder/{folder_id}/list"
    data = {
        "name": name
    }
    response = requests.post(url, headers=HEADERS, json=data)
    return response.json()

def create_task(list_id, name, description,priority,assignee_name):
    assignee_id = user_ids.get(assignee_name)  # Get the user ID for the assignee
    if not assignee_id:
        return {"error": "User not found"}
    url = f"{BASE_URL}list/{list_id}/task"
    data = {
        "name": name,
        "description": description,
        "priority":priority,
        "assignees": [assignee_id]
    }
    response = requests.post(url, headers=HEADERS, json=data)
    return response.json()

@app.route('/create_structure', methods=['POST'])
def create_structure():
    data = request.json
    folder_name = data.get('folderName')
    list_name = data.get('listName')
    task_name = data.get('taskName')
    description = data.get('description')
    priority = data.get('priority')
    assignee_name = data.get("assignee_name")
    space_id = '3389419'  # Assuming you have a space_id
    
    folder_response = create_folder(space_id, folder_name)
    list_response = create_list(folder_response.get('id'), list_name)
    task_response = create_task(list_response.get('id'), task_name, description, priority, assignee_name)
    
    return jsonify(response=task_response, list_id=list_response.get('id'))

user_ids = {
    "Muaz Ashraf": 44183335,
    "Ian Matenaer": 44183332,
    "Saifullah Sohail": 72038349,
    "Jostens": 38130033,
    "ASID User": 38114073,
    "Guest User": 26394280,
    "Beau Gilles": 38105691,
    "Info Test": 32332725,
    "Henry Hoeglund": 14933310,
    "Davyd Barchuk": 32161898,
    "Cooper McKinnon": 32161897,
    "Andy Rose": 32161896,
    "Chris Handrick": 12853715,
    "Beau Gilles": 10760744,  # Note: This user is listed twice. This ID will override the previous one.
    "Nick Michael": 10760743,
    "Zain Zulifqar": 10760740,
    "Maggie Heilmann": 10760733,
    "McKinzie Plots": 10760732,
    "Christopher Benjamin": 10760731,
    "Kira Diner": 10684981,
    "Klaus Winkler": 6619186,
    "Kristine Matenaer": 10666395,
    "anna matenaer": 10520440,
    "David T Matenaer": 10518265
}
@app.route('/get_names', methods=['GET'])
def get_names():
    return jsonify(names=list(user_ids.keys()))



if __name__ == '__main__':
    app.run(debug=True)
