from typing import List
from langchain_core.documents import Document
from ddgs import DDGS


# Direct DuckDuckGo search fallback returning LangChain Document objects.
# Compatible with both `ddgs` and `duckduckgo_search` libraries.
def search_web_fallback(query: str, max_results: int = 4) -> List[Document]:
    try:
        results = []
        with DDGS() as ddgs:
            ddg_gen = ddgs.text(query, max_results=max_results)
            for r in ddg_gen:
                title = r.get("title", "Web Source")
                body = r.get("body", "")
                href = r.get("href", "")

                if body:
                    results.append(
                        Document(
                            page_content=body,
                            metadata={
                                "source_file": title,
                                "url": href,
                                "source_type": "web",
                            },
                        )
                    )
        return results
    except Exception as e:
        print(f"⚠️ Web search fallback error: {e}")
        return []
