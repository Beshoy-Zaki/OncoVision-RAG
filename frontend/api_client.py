import os
import httpx
from dotenv import load_dotenv

# Load frontend environment
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

class BackendAPIClient:
    def __init__(self, base_url: str = None, timeout: float = 90.0):
        self._base_url = base_url
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        if self._base_url:
            return self._base_url
        load_dotenv(override=True)
        return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

    def check_health(self) -> dict:
        """Query the /health endpoint to verify backend status."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/health")
                if res.status_code == 200:
                    return res.json()
        except Exception:
            pass
        return None

    def query(self, question: str, image_bytes: bytes = None, filename: str = "scan.jpg") -> dict:
        """Send a question with optional MRI image scan to POST /query."""
        url = f"{self.base_url}/query"
        data = {"question": question}
        files = None
        
        if image_bytes:
            files = {"file": (filename, image_bytes, "image/jpeg")}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, data=data, files=files)
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": True,
                        "status_code": response.status_code,
                        "detail": response.text
                    }
        except httpx.ConnectError:
            return {
                "error": True,
                "detail": f"Could not connect to backend at {self.base_url}. Is FastAPI running on port 8000?"
            }
        except Exception as e:
            return {
                "error": True,
                "detail": str(e)
            }

api_client = BackendAPIClient()
