from src.routers import router
from dotenv import load_dotenv
load_dotenv(override=True)

if __name__ == '__main__':
    query = "What is the summary of the page 2"
    router(query=query)
