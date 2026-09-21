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

# Ensure ML model is copied (if exists)
COPY ["ML/model.onnx", "ML/model.onnx"]
# Note: Add your model.onnx to ML/ folder before deploying

# Expose port (Render sets PORT env var)
EXPOSE 8080

# Start
ENTRYPOINT ["dotnet", "PerceptAI.API.dll"]