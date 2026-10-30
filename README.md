# RESUME-# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An **ATS score** out of 100
- Scores for 5 categories (formatting, keywords, impact, education, clarity)
- Found and missing keywords (optionally matched against a job description)
- Prioritised improvements (High / Medium / Low)
- Example rewrites of weak bullet points
- Automatic checks (email, phone, LinkedIn, standard sections, word count)

Built with **Streamlit** and **Google Gemini Flash**.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GEMINI_API_KEY="your_key"   # Windows PowerShell: $env:GEMINI_API_KEY="your_key"
streamlit run app.py
```

Get a free API key at https://aistudio.google.com/apikey. You can also paste the key into the app's sidebar instead of setting the variable.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (never commit your API key).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app** → select your repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your_key"
   ```
5. Click **Deploy**.

## Configuration

| Item | Where | Notes |
|------|-------|-------|
| API key | `GEMINI_API_KEY` in Streamlit secrets or env var | Or type it into the sidebar |
| Model | Sidebar text box / `DEFAULT_MODEL` in `app.py` | Any Gemini Flash model name. If it is rejected, the app falls back to others |

## Limitations

- Scanned (image-only) PDFs can't be read; use a text-based PDF or DOCX.
- The score is an AI estimate, not the output of a real employer's ATS.
- Resume text is sent to Google's Gemini API for analysis. Nothing is stored by the app.
