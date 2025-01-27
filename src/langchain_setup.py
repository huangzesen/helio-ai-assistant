import re
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class ClassifierChain(LLMChain):
    def __init__(self):
        llm = OllamaLLM(model="deepseek-r1:7b")
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template=(
                "Does query start with 'search' or 'look for'? Please answer with 'yes' or 'no'. If not sure, answer 'no'. "
                "If 'yes', extract the main key concept from the input suitable for a research paper search. If 'no', the key concept is 'None'.\n"
                "Format the output as follows:\n"
                "Classification: yes|no\n"
                "Key-Concept: concept\n"
                "Query: {query}"
            )
        )
        super().__init__(llm=llm, prompt=prompt_template, output_key="classification")

    def _call(self, inputs):
        query = inputs["query"]
        prompt = self.prompt.format(query=query)
        response = self.llm(prompt)
        
        if response:
            # Print the raw response for debugging
            print("Raw response from LLM:")
            print(response)
            
            # Remove the <think> block
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

            # Extract the final answer and key concept
            classification_match = re.search(r'Classification:\s*(yes|no)', response, re.IGNORECASE)
            key_concept_match = re.search(r'Key-Concept:\s*(.*)', response, re.IGNORECASE)
            if classification_match and key_concept_match:
                classification = classification_match.group(1).lower()
                key_concept = key_concept_match.group(1).strip()
                if key_concept.lower() == "none":
                    key_concept = ""
                return {"classification": classification == 'yes', "keyword": key_concept}
        
        return {"classification": False, "keyword": ""}

class LocalModelChain(LLMChain):
    def __init__(self):
        llm = OllamaLLM(model="deepseek-r1:14b")
        memory = ConversationBufferMemory()
        prompt_template = PromptTemplate(input_variables=["query", "history"], template="{history}\nUser: {query}")
        super().__init__(llm=llm, prompt=prompt_template, memory=memory, output_key="response")

    def _call(self, inputs):
        query = inputs["query"]
        history = self.memory.load_memory_variables(inputs)
        conversation = self.prompt.format(query=query, history="\n".join(history))
        response = self.llm(conversation)
        
        if response:
            # Remove the <think> block
            primary_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            self.memory.save_context({"User": query}, {"Bot": primary_response})
            return {"response": primary_response}
        
        return {"response": None}

import unittest
from langchain_setup import ClassifierChain
from nasa_ads import search_nasa_ads

class TestNasaAds(unittest.TestCase):
    def test_search_nasa_ads(self):
        query = "solar wind turbulence"
        result = search_nasa_ads(query)
        
        self.assertIsNotNone(result, "The result should not be None")
        self.assertIn("response", result, "The result should contain 'response'")
        self.assertIn("docs", result["response"], "The 'response' should contain 'docs'")
        self.assertGreater(len(result["response"]["docs"]), 0, "There should be at least one document in the response")
        
        # Print the query results
        print("Query Results:")
        for doc in result["response"]["docs"]:
            title = doc.get("title", ["No title"])[0]
            authors = ", ".join(doc.get("author", ["No authors"]))
            print(f"Title: {title}\nAuthors: {authors}\n")
        
        # Check the first document for expected fields
        doc = result["response"]["docs"][0]
        self.assertIn("title", doc, "The document should contain 'title'")
        self.assertIn("author", doc, "The document should contain 'author'")
        # Check if abstract is present
        if "abstract" in doc:
            self.assertIn("abstract", doc, "The document should contain 'abstract'")

    def test_classifier_chain(self):
        classifier_chain = ClassifierChain()
        query = "Tell me about the solar wind"
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