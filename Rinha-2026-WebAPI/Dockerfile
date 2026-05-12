# ====== Stage 1: Train LightGBM model (Python) ======
FROM python:3.12-slim AS train
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train_model.py .
COPY data/references.json.gz /data/references.json.gz
RUN python train_model.py /data/references.json.gz /data/model.bin

# ====== Stage 2: Build IVF index (C# Preprocessor) ======
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS preprocess
RUN apt-get update && apt-get install -y --no-install-recommends clang zlib1g-dev
WORKDIR /src
COPY Rinha-2026-Preprocessor/Rinha-2026-Preprocessor.csproj Rinha-2026-Preprocessor/
RUN dotnet restore Rinha-2026-Preprocessor/Rinha-2026-Preprocessor.csproj
COPY Rinha-2026-Preprocessor/ Rinha-2026-Preprocessor/
WORKDIR /src/Rinha-2026-Preprocessor
RUN dotnet publish -c Release -o /app/preprocess
COPY data/references.json.gz /data/references.json.gz
RUN /app/preprocess/Rinha-2026-Preprocessor /data/references.json.gz /data/index.bin

# ====== Stage 3: Build API (C# AOT) ======
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS publish
RUN apt-get update && apt-get install -y --no-install-recommends clang zlib1g-dev
WORKDIR /src
COPY Rinha-2026-WebAPI/Rinha-2026-WebAPI.csproj Rinha-2026-WebAPI/
RUN dotnet restore Rinha-2026-WebAPI/Rinha-2026-WebAPI.csproj
COPY Rinha-2026-WebAPI/ Rinha-2026-WebAPI/
WORKDIR /src/Rinha-2026-WebAPI
RUN dotnet publish -c Release -o /app/publish /p:UseAppHost=true

# ====== Stage 4: Runtime ======
FROM mcr.microsoft.com/dotnet/runtime-deps:10.0 AS final
WORKDIR /app
EXPOSE 8080
COPY --from=publish /app/publish .
COPY --from=train /data/model.bin /app/model.bin
COPY --from=preprocess /data/index.bin /app/index.bin
ENV INDEX_PATH=/app/index.bin
ENV MODEL_PATH=/app/model.bin
ENTRYPOINT ["./Rinha-2026-WebAPI"]