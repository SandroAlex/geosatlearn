# Load packages.
from fastapi import FastAPI

app = FastAPI()


@app.get("/", include_in_schema=True)
def read_root():
    return {"message": "Hello, World!"}
