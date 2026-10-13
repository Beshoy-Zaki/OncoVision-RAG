import json
import mimetypes
import os
import httpx
from dotenv import load_dotenv

load_dotenv()


class BackendAPIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 90.0):
        self._base_url = base_url
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url
        load_dotenv(override=True)
        return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

    def check_health(self) -> dict | None:
        """Query the /health endpoint to verify backend status."""
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    def query(
        self,
        question: str,
        image_bytes: bytes | None = None,
        filename: str = "scan.jpg",
        chat_history: list | None = None,
    ) -> dict:
        """Send a clinical question with optional MRI scan and chat history to POST /query."""
        url = f"{self.base_url}/query"
        data = {"question": question}
        if chat_history:
            data["chat_history_json"] = json.dumps(chat_history[-4:])

        files = None
        if image_bytes:
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "image/jpeg"
            files = {"file": (filename, image_bytes, mime_type)}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, data=data, files=files)
                if response.status_code == 200:
                    return response.json()
                try:
                    err_json = response.json()
                    detail = err_json.get("detail", response.text)
                except Exception:
                    detail = response.text
                return {
                    "error": True,
                    "status_code": response.status_code,
                    "detail": detail,
                }
        except httpx.ConnectError:
            return {
                "error": True,
                "detail": f"Could not connect to backend at {self.base_url}. Ensure FastAPI is running on port 8000.",
            }
        except httpx.ReadTimeout:
            return {
                "error": True,
                "detail": "Request timed out while waiting for the RAG pipeline.",
            }
        except Exception as e:
            return {"error": True, "detail": str(e)}


api_client = BackendAPIClient()
