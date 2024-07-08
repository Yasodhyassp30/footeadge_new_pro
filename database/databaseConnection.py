from pymongo import MongoClient
import os
from dotenv import load_dotenv

from utils.logger import configure_logger

logger = configure_logger()

load_dotenv()

def get_db():
    URI = os.getenv('MONGO_URI')
    client = MongoClient('mongodb+srv://Malaka:fyp2024@cluster0.wzyxz42.mongodb.net/football_analysis?retryWrites=true&w=majority&appName=Cluster0')
    return client.football_analysis
