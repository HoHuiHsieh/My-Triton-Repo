FROM nvcr.io/nvidia/tritonserver:25.04-py3
# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-openai \
    python3-numpy \
    python3-pandas \
    python3-sklearn \
    python3-sklearn-lib \
    python-sklearn-doc \
    python3-scipy \
    python3-matplotlib \
    python3-seaborn \
    python3-psycopg2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U \
    langgraph \
    langchain_community \
    langchain-text-splitters \
    langchain_openai \
    langchain_postgres \
    --ignore-installed \
    --break-system-packages \
    && rm -rf /root/.cache/pip
