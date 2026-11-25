# Insurance Policy RAG Chatbot

A terminal chatbot that indexes local insurance policy PDFs and TXT files with
FastEmbed and Qdrant Local, retrieves relevant excerpts, and asks Groq for an
answer. Embeddings and the vector database run locally; LLM requests are sent
to Groq.

## Setup (Windows PowerShell)

From this directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add your Groq API key to `.env`:

```text
GROQ_API_KEY=your_actual_key
```

Do not share or commit `.env`. If Windows Application Control blocks package
installation, use an installation method approved by your organization.

## Add policies and run

Put text-based PDFs or `.txt` files in `data\`, then run:

```powershell
python main.py
```

The included `data\demo_policy.txt` can be used to try the app. Scanned PDFs
that contain no selectable text need OCR before they can be indexed.

The app recreates the `insurance_policies` collection on each start so the
local index reflects the current files in `data\`. The database is stored in
`qdrant_local\`. Type `exit` or `quit` to end the chat.
