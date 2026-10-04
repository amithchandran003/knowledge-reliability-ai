old_document = {
    "title": "Data Privacy Policy",
    "source": "Internal Compliance",
    "version": "1.2",
    "updated_date": "2025-01-15",
    "content": "All personal data must be encrypted at rest and in transit."
}
new_document = {
    "title": "Data Privacy Policy",
    "source": "Internal Compliance",
    "version": "1.3",
    "updated_date": "2025-02-15",
    "content": "All personal data must be encrypted using AES-256."
}
def compare_document(old_document,new_document):
    change={}
    old_word=old_document.split(" ")
    new_word=new_document.split(" ")
    for word in new_word:
        if word not in old_word:
            print(word)
    #     if old_document["title"]!=new_document["title"]:
    #         change["title"]={
    #             "old":old_document["title"],
    #             "new":new_document["title"]
    #         }
    #     else:
    #         print("title : unchanged")
    # if old_document["source"]!=new_document["source"]:
    #     change["source"]={
    #                    "old":old_document["source"],
    #                    "new":new_document["source"]
    #             }
    # else:
    #           print("source : unchanged")
    # if old_document["version"]!=new_document["version"]:
    #         change["version"]={
    #                        "old":old_document["version"],
    #                        "new":new_document["version"]}
    # else:
    #         print("version : unchanged")
    # if old_document["updated_date"]!=new_document["updated_date"]:
    #         change["updated_date"]={
    #                        "old":old_document["updated_date"],
    #                        "new":new_document["updated_date"]
    #                 }
    # else:
    #           print("ti : unchanged")
    # if old_document["content"]!=new_document["content"]:
    #            change["content"]={
    #                                       "old":old_document["content"],
    #                                       "new":new_document["content"]
    #                                }
    # else:
    #        print("content : unchanged")
    # return change
changes = compare_document(old_document, new_document)
print(changes)
    