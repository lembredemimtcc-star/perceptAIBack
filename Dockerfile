# Build stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY ["PerceptAI.API.csproj", "./"]
RUN dotnet restore "PerceptAI.API.csproj"

# Copy everything and build
COPY . .
RUN dotnet publish "PerceptAI.API.csproj" -c Release -o /app/publish --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app

# Copy published app
COPY --from=build /app/publish .

# Copy ML model if it exists (using shell to avoid build failure when file is missing)
RUN mkdir -p ML && [ -f /src/ML/model.onnx ] && cp /src/ML/model.onnx ML/model.onnx || true
# Note: Add your model.onnx to ML/ folder before deploying, or mount it at runtime

# Expose port (Render sets PORT env var)
EXPOSE 8080

# Start
ENTRYPOINT ["dotnet", "PerceptAI.API.dll"]