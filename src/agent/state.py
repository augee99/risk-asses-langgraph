from typing import List, Dict, Any, Optional

class State:
    """State class to manage the workflow state."""
    
    def __init__(self, messages: Optional[List[Dict[str, str]]] = None, pdf_chunks: Optional[List[Any]] = None):
        self.messages = messages or []
        self.pdf_chunks = pdf_chunks or []

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state by key."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"State(messages={self.messages}, pdf_chunks={self.pdf_chunks})"
