import os, json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os, json

load_dotenv()  # ✅ Loads variables from .env file


# load JSON string from env (or point to file path)
json_input = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
doc_id = os.getenv("GOOGLE_DOC_ID")

if not json_input or not doc_id:
    raise SystemExit("Set GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_DOC_ID")

# load credentials (either JSON string or path)
try:
    if os.path.exists(json_input):
        creds = service_account.Credentials.from_service_account_file(
            json_input, scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
    else:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(json_input),
            scopes=["https://www.googleapis.com/auth/documents.readonly"]
        )
except Exception as e:
    raise SystemExit("Failed to load credentials: " + str(e))

service = build("docs", "v1", credentials=creds, cache_discovery=False)
doc = service.documents().get(documentId=doc_id).execute()
print("Title:", doc.get("title"))
# print first 500 chars of the doc body for sanity
body = doc.get("body", {}).get("content", [])
flat_text = []
for el in body:
    if "paragraph" in el:
        for e in el["paragraph"].get("elements", []):
            tr = e.get("textRun")
            if tr and "content" in tr:
                flat_text.append(tr["content"])
print("Excerpt:", "".join(flat_text)[:500])
