
from utils.dataclass import RetrievedDocument


class Retriever:
    def __init__(self):
        pass

    def aretrieve(self, query: str, **kwargs) -> list[RetrievedDocument]:
        raise NotImplementedError("Subclasses must implement the aretrieve method.")
