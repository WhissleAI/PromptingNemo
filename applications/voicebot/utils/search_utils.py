from duckduckgo_search import DDGS

def search_duckduckgo(query, max_results=2):
    results = DDGS().text(query, max_results=max_results)
    urls = [result['href'] for result in results]
    return urls

