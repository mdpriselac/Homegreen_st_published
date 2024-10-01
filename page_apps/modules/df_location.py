import streamlit as st
import boto3

aws_access = st.secrets["amzn"]["aws_axs"]
aws_secret = st.secrets["amzn"]["aws_secret"]
bucket = st.secrets["db"]["bucket"]
db_path = st.secrets["db"]["db_path"]


s3_client = boto3.client('s3', aws_access_key_id=aws_access, aws_secret_access_key=aws_secret)



