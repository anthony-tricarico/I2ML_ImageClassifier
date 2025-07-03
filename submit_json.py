import json
import requests
import os

def submit_from_json(json_file_path, groupname, url="http://tatooine.disi.unitn.it:3001/retrieval/"):
    """
    Submit results from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing results
        groupname: Your group name for submission
        url: Submission endpoint URL
    """
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"ERROR: File {json_file_path} not found!")
        return
    
    # Load results from JSON file
    try:
        with open(json_file_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded results from {json_file_path}")
        print(f"Number of queries: {len(results)}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format in {json_file_path}: {e}")
        return
    except Exception as e:
        print(f"ERROR: Could not read file {json_file_path}: {e}")
        return
    
    # Prepare submission payload
    res = {
        "groupname": groupname,
        "images": results
    }
    
    # Submit results
    try:
        print("Submitting results...")
        response = requests.post(url, json=res)
        
        # Parse response
        try:
            result = response.json()
            print(f"SUCCESS: Accuracy is {result['accuracy']}")
        except json.JSONDecodeError:
            print(f"ERROR: Invalid response format: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Network error during submission: {e}")

# Usage: Submit from JSON file in root directory
json_filename = "TL_6_Epo1.json"  # Change this to your JSON filename
submit_from_json(json_filename, groupname="Overfit & Underpaid")

# Alternative: If you want to specify the full path
# submit_from_json("/path/to/your/results.json", groupname="Overfit & Underpaid")