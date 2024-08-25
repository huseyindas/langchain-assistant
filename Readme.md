# LangChain Assistant

This repository contains a LangChain-based assistant built using FastAPI, LangChain, and Retrieval-Augmented Generation (RAG). The assistant can load data from sitemaps and PDFs, vectorize it, and store it in a database for intelligent querying. It also provides a chat interface for interacting with the bot.

## Features
- **Health Endpoint:** Monitor the health of the service.
- **Chat Endpoint (`/bot/chat`):** Interact with the assistant in real-time.
- **Load Endpoint (`/bot/load`):** Vectorize and store data from sitemaps and PDFs.
- **Custom Prompt Support:** The assistant uses a prompt defined in the `prompt.txt` file located in the root directory.

## Setup and Installation

### Prerequisites
- Docker
- Docker Compose

### Environment Variables
Create an `.env` file in the root directory or use the provided `example.env` file to set up your environment variables.

### Docker Setup
Build and run the application using Docker Compose:

```bash
docker-compose up --build
