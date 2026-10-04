# policy_documents.py
import json
import os


class Document:
    """Represents a policy document."""

    def __init__(self, title, source, version, updated_date, content):
        self.title = title
        self.source = source
        self.version = version
        self.updated_date = updated_date
        self.content = content


def load_policies():
    """Opens data/policies.json, reads data, and returns a list of Document objects."""
    # Find data/policies.json relative to this script
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "data", "policies.json")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Convert each dictionary into a Document object
    documents = []
    for item in data:
        doc = Document(
            title=item["title"],
            source=item["source"],
            version=item["version"],
            updated_date=item["updated_date"],
            content=item["content"],
        )
        documents.append(doc)

    return documents


def show_document(document):
    """Prints all fields of a single document."""
    print(f"Title: {document.title}")
    print(f"Source: {document.source}")
    print(f"Version: {document.version}")
    print(f"Updated Date: {document.updated_date}")
    print(f"Content: {document.content}")


def show_all_documents(documents):
    """Loops through all documents and displays each with a separator."""
    print("-" * 50)
    for doc in documents:
        show_document(doc)
        print("-" * 50)


if __name__ == "__main__":
    # Load policies as Document objects and display them
    policies = load_policies()
    show_all_documents(policies)
