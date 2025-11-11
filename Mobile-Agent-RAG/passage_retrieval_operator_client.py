import requests
import os
from datetime import datetime
import json

def download_images(
    docs,
    server_url="http://localhost:8002",
    base_save_dir="downloaded_images"
):
    # Generate folder name with current time (up to seconds)
    timestamp_folder = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join(base_save_dir, timestamp_folder)
    os.makedirs(save_dir, exist_ok=True)

    for i, doc in enumerate(docs, 1):
        url = server_url + doc["image_url"]  # e.g. http://.../images/xxx.png
        filename = os.path.basename(doc["image_url"])
        local_path = os.path.join(save_dir, filename)
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            # Update image_url to local absolute path
            doc["image_url"] = os.path.abspath(local_path)
        except Exception as e:
            print(f"[{i}/{len(docs)}] Download failed: {e}")
    
    return save_dir  # Return save path for higher-level use

def query_operator_server(query: str, server_url: str = "http://localhost:8002", n_docs: int = 1, save_images=True, base_save_dir="downloaded_images"):
    # Step 1: Query the server
    resp = requests.post(f"{server_url}/retrieve", json={"query": query, "n_docs": n_docs}, timeout=60)
    resp.raise_for_status()
    results = resp.json()["results"]

    # Step 2: Whether to save images
    if save_images:
        save_path = download_images(results, server_url=server_url, base_save_dir=base_save_dir)
        # print(f"Images saved to: {save_path}")

    return results

def upload_action_to_server(last_action, instruction, screenshot_path, server_url="http://localhost:8002"):
    action_name = last_action.get("name", "")
    action_args = last_action.get("arguments", {})
    action_text = f"{action_name} at {json.dumps(action_args)}"

    data = {
        "instruction": instruction,
        "action_text": action_text
    }
    files = {
        "screenshot": (os.path.basename(screenshot_path), open(screenshot_path, "rb"), "image/png")
    }
    resp = requests.post(f"{server_url}/append_tsv", data=data, files=files, timeout=60)
    resp.raise_for_status()
    print("Uploaded, new id:", resp.json().get("id"))

if __name__ == "__main__":
    # 示例：上传 last_action

    docs = query_operator_server("Subgoal: Open X. App: X", base_save_dir="./youTube_images")
    print("Retrieval Results:", docs)

    last_action = {"name": "Tap", "arguments": {"x": 100, "y": 200}}
    instruction = "Open X. App: X"
    screenshot_path = "./screenshot/screenshot.jpg"  # 当前截图路径
    upload_action_to_server(last_action, instruction, screenshot_path)