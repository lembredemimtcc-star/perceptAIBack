using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using PerceptAI.API.ML;
using PerceptAI.API.Services;

var builder = WebApplication.CreateBuilder(args);

// =====================================================================
// REGISTRO DE SERVIÇOS DO CONTAINER DI
// =====================================================================

builder.Services.AddControllers();

// Configuração do CORS para permitir que o front-end em React Native (Expo)
// possa efetuar requisições sem restrições de domínios cruzados localmente.
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

// 1. Registrar ImagePreprocessingService como Singleton
builder.Services.AddSingleton<ImagePreprocessingService>();

// 2. Registrar ModelLoader como Singleton (garante carregamento único do modelo)
builder.Services.AddSingleton(sp =>
{
    // Resolve o caminho do modelo ONNX na pasta ML
    var modelPath = Path.Combine(AppContext.BaseDirectory, "ML", "model.onnx");
    if (!File.Exists(modelPath))
    {
        modelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "model.onnx");
    }

    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException(
            $"O modelo ONNX não foi encontrado no caminho especificado: '{modelPath}'. " +
            "Certifique-se de executar o script Python 'export_onnx.py' antes de rodar a API.");
    }

    return new ModelLoader(modelPath);
});

// 3. Registrar EmotionDetectionService como Singleton
builder.Services.AddSingleton<EmotionDetectionService>();

var app = builder.Build();

// =====================================================================
// PIPELINE DE REQUISIÇÕES HTTP (MIDDLEWARES)
// =====================================================================

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

// Aplica a política CORS
app.UseCors("AllowAll");

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
