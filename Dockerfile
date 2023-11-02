FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer;model = SentenceTransformer('all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1

COPY . .

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "wsgi:app"]