# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY ["PerceptAI.API.csproj", "./"]
RUN dotnet restore "PerceptAI.API.csproj"

# Copy everything and build
COPY . .
RUN dotnet publish "PerceptAI.API.csproj" -c Release -o /app/publish --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final
WORKDIR /app

# Copy published app
COPY --from=build /app/publish .

# Copy ML model and data files from the build stage (where /src exists)
COPY --from=build /src/ML ./ML/

# Expose port (Render sets PORT env var)
EXPOSE 8080

# Start
ENTRYPOINT ["dotnet", "PerceptAI.API.dll"]