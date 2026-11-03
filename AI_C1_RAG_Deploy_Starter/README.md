# AI-C1 Document RAG

1. Copy your existing PDF and TXT files directly into `real_estate_data/` (not subfolders).
2. Create a GitHub repository and upload these starter files plus those documents. Do not upload API keys.
3. In Render, create a Web Service from that repo. Use the included render.yaml settings.
4. In Render Environment settings, set `GEMINI_API_KEY` to your secret key.
5. Deploy; first startup downloads the embedding model and indexes the documents and may take several minutes.
6. Test `/health`, then ask a question that only a document can answer.
Text-based PDFs and TXT files are supported. Scanned PDFs need OCR. Local Chroma data is generated from the included documents during startup.
