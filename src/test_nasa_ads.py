import unittest
from langchain_setup import ClassifierChain
from nasa_ads import search_nasa_ads

class TestNasaAds(unittest.TestCase):
    # def test_search_nasa_ads(self):
    #     query = "solar wind turbulence"
    #     result = search_nasa_ads(query)
        
    #     self.assertIsNotNone(result, "The result should not be None")
    #     self.assertIn("response", result, "The result should contain 'response'")
    #     self.assertIn("docs", result["response"], "The 'response' should contain 'docs'")
    #     self.assertGreater(len(result["response"]["docs"]), 0, "There should be at least one document in the response")
        
    #     # Print the query results
    #     print("Query Results:")
    #     for doc in result["response"]["docs"]:
    #         title = doc.get("title", ["No title"])[0]
    #         authors = ", ".join(doc.get("author", ["No authors"]))
    #         print(f"Title: {title}\nAuthors: {authors}\n")
        
    #     # Check the first document for expected fields
    #     doc = result["response"]["docs"][0]
    #     self.assertIn("title", doc, "The document should contain 'title'")
    #     self.assertIn("author", doc, "The document should contain 'author'")
    #     # Check if abstract is present
    #     if "abstract" in doc:
    #         self.assertIn("abstract", doc, "The document should contain 'abstract'")

    def test_classifier_chain(self):
        classifier_chain = ClassifierChain()
        query = "Tell me about the solar wind turbulence"
        result = classifier_chain.invoke({"query": query})
        
        # Print the classifier chain response
        print("Classifier Chain Response:")
        print(result)
        
        self.assertIn("classification", result, "The result should contain 'classification'")
        self.assertIn("keyword", result, "The result should contain 'keyword'")
        self.assertIsInstance(result["keyword"], str, "Keyword should be a string")
        self.assertGreater(len(result["keyword"]), 0, "There should be a keyword")

if __name__ == "__main__":
    unittest.main()