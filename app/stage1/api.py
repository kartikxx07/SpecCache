import fastapi
import requests

url = "http://localhost:8000/endpoint"

app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/data")
def data():
    return get_data_from_endpoint()


def get_data_from_endpoint():
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"response": {"user_response": {"error": str(e)}}}