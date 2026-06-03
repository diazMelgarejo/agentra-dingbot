FROM python:3.11-slim

# System dependencies for TA-Lib C library
RUN apt-get update && apt-get install -y gcc make wget && rm -rf /var/lib/apt/lists/*

RUN wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . 2>/dev/null || true

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python"]
