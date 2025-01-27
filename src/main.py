from langchain_setup import ClassifierChain, LocalModelChain
from nasa_ads import search_nasa_ads

def main():
    print("Welcome to my Python project!")
    classifier_chain = ClassifierChain()
    local_model_chain = LocalModelChain()

    while True:
        user_input = input("Please enter your query (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Run the classifier chain
        classification_result = classifier_chain.invoke({"query": user_input})
        is_nasa_ads = classification_result["classification"]
        keyword = classification_result["keyword"]
        
        if is_nasa_ads:
            print(f"Querying NASA ADS with keyword: {keyword}")
            search_results = search_nasa_ads(keyword)
            if search_results:
                for doc in search_results.get("response", {}).get("docs", []):
                    title = doc.get("title", ["No title"])[0]
                    authors = ", ".join(doc.get("author", ["No authors"]))
                    abstract = doc.get("abstract", "No abstract")
                    print(f"Title: {title}\nAuthors: {authors}\nAbstract: {abstract}\n")
        else:
            # Run the local model chain
            response_result = local_model_chain.invoke({"query": user_input})
            response = response_result["response"]
            if response:
                print(f"Local model response: {response}")

if __name__ == "__main__":
    main()