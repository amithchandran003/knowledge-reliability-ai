# api_test.py
import json
import ssl
import urllib.request

url = "https://jsonplaceholder.typicode.com/todos/1"

# Create an SSL context to bypass certificate verification issues on macOS
context = ssl._create_unverified_context()

# Make the GET request
with urllib.request.urlopen(url, context=context) as response:
    # Print the HTTP status code
    print(f"Status Code: {response.status}")

    # Read the response and parse it as JSON
    response_body = response.read().decode("utf-8")
    data = json.loads(response_body)

    # Print the returned JSON data
    print("Response Data:")
    print(data)
