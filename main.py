# from typing import Optional

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from llama3 import app

App = FastAPI(title="Dynamic chatbot App")

# Set all CORS enabled origins
App.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(App, app)

# if __name__ == "__main__":
# uvicorn.run(App, host="localhost", port=8500)