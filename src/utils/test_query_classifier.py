from query_classifier import is_nasa_ads_query

def test_classifier():
    test_query = "query nasa ads blablabla"
    result = is_nasa_ads_query(test_query, verbose=True)
    print(f"Query: {test_query}")
    print(f"Is NASA ADS query: {result}")

if __name__ == "__main__":
    test_classifier()