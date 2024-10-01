import pandas as pd
import chromadb
import streamlit as st
import boto3
from page_apps.modules.similarity_db_tools import create_full_similarity_db, create_flavor_similarity_db

### Step 1: Get the AWS Data Loaded
#### Step 1: Establish AWS Credentials
aws_access = st.secrets["amzn"]["aws_axs"]
aws_secret = st.secrets["amzn"]["aws_secret"]
bucket = st.secrets["db"]["bucket"]
db_path = st.secrets["db"]["db_path"]

s3_client = boto3.client('s3', aws_access_key_id=aws_access, aws_secret_access_key=aws_secret)

#### Step 2: Load the Full Data Set
@st.cache_data
def load_full_dataset():
    obj = s3_client.get_object(Bucket=bucket, Key=db_path)
    in_df = pd.read_csv(obj['Body'],index_col=0)
    out_df = in_df.copy()
    return out_df


### Step 2: Create two Chroma DB connections
#### Step 2a: Create the Chroma DB Client and Connections
client = chromadb.Client()

@st.cache_resource
def create_flavor_connection():
    return client.get_or_create_collection('flavor_similarity_db',metadata={"hnsw:space":"cosine"})

@st.cache_resource
def create_full_connection():
    return client.get_or_create_collection('full_similarity_db',metadata={"hnsw:space":"cosine"})    

### Step 3: Populate the Databases
flav_only_conn = create_flavor_connection()
full_conn = create_full_connection()
#### Step 3a: Create the dataframe
raw_loaded_df = load_full_dataset()
#### Create the full similarity Database
create_full_similarity_db(raw_loaded_df,full_coll_conn=full_conn)
#### Create the Flavor Only Similarity Database
create_flavor_similarity_db(raw_loaded_df,flav_coll_conn=flav_only_conn)