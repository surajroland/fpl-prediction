FROM ubuntu:24.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    build-essential cmake \
    git curl wget \
    htop nano vim tree \
    ca-certificates \
    sudo \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Install starship prompt
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes
RUN echo 'eval "$(starship init bash)"' >> ~/.bashrc

# Switch to user
USER root

# Set working directory
WORKDIR /workspace

# Install core ML runtime
RUN pip install --no-cache-dir --break-system-packages \
    xgboost==2.1.2 \
    pandas==2.2.3 \
    numpy==2.0.2 \
    scikit-learn==1.5.2

# Copy requirements and install dev dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy and install project
COPY . .
RUN pip install --break-system-packages -e .

# Expose ports
EXPOSE 8888 6006 5000

CMD ["bash"]
