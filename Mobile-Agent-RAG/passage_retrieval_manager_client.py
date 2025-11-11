import requests

def query_manager_server(query: str, server_url: str = "http://localhost:8000", n_docs: int = 3):
    payload = {
        "query": query,
        "n_docs": n_docs
    }
    
    try:
        response = requests.post(
            f"{server_url}/retrieve",
            json=payload,
            timeout=30  
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    test_query = "Search for learn spanish for beginners on YouTube, filter by this month, and add the first video to the Watch Later playlist"
    result = query_manager_server(test_query)
    print("Retrieval Results:", result)