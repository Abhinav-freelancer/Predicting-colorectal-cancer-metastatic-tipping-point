FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    streamlit plotly fastapi uvicorn \
    pandas numpy scipy scikit-learn \
    torch --index-url https://download.pytorch.org/whl/cpu \
    shap pyyaml

COPY . .

EXPOSE 8501 8000

CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
