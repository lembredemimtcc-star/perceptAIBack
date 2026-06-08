using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using PerceptAI.API.ML;
using PerceptAI.API.Services;
using PerceptAI.API.Settings;

var builder = WebApplication.CreateBuilder(args);

// =====================================================================
// CONFIGURAÇÕES
// =====================================================================

// Vincula a seção "SupabaseSettings" do appsettings (ou variáveis de ambiente
// no formato SupabaseSettings__Url / SupabaseSettings__ServiceRoleKey)
// ao record tipado SupabaseSettings.
builder.Services.Configure<SupabaseSettings>(
    builder.Configuration.GetSection(SupabaseSettings.SectionName));

// =====================================================================
// REGISTRO DE SERVIÇOS
// =====================================================================

builder.Services.AddControllers();

// CORS — permite que o front-end Expo possa fazer requisições locais
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

builder.Services.AddOpenApi();

// ------------------------------------------------------------------
// HttpClient do Supabase (pré-configurado com URL base + headers auth)
// ------------------------------------------------------------------
builder.Services.AddHttpClient("supabase", (sp, client) =>
{
    var settings = sp.GetRequiredService<IOptions<SupabaseSettings>>().Value;

    if (string.IsNullOrWhiteSpace(settings.Url))
        throw new InvalidOperationException(
            "SupabaseSettings:Url não configurado. " +
            "Defina em appsettings.Development.json ou na variável de ambiente SupabaseSettings__Url.");

    if (string.IsNullOrWhiteSpace(settings.ServiceRoleKey))
        throw new InvalidOperationException(
            "SupabaseSettings:ServiceRoleKey não configurado. " +
            "Defina em appsettings.Development.json ou na variável de ambiente SupabaseSettings__ServiceRoleKey.");

    // Base URL aponta para a REST API PostgREST do Supabase
    client.BaseAddress = new Uri($"{settings.Url.TrimEnd('/')}/rest/v1/");

    // Headers obrigatórios para autenticação com Service Role Key
    client.DefaultRequestHeaders.Add("apikey", settings.ServiceRoleKey);
    client.DefaultRequestHeaders.Add("Authorization", $"Bearer {settings.ServiceRoleKey}");

    // Garante que as respostas JSON sejam aceitas
    client.DefaultRequestHeaders.Add("Accept", "application/json");
});

// ------------------------------------------------------------------
// Serviços de negócio
// ------------------------------------------------------------------

// 1. Pré-processamento de imagem
builder.Services.AddSingleton<ImagePreprocessingService>();

// 2. Modelo ONNX (carregado uma única vez como Singleton)
builder.Services.AddSingleton(sp =>
{
    var modelPath = Path.Combine(AppContext.BaseDirectory, "ML", "model.onnx");
    if (!File.Exists(modelPath))
        modelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "model.onnx");

    if (!File.Exists(modelPath))
        throw new FileNotFoundException(
            $"O modelo ONNX não foi encontrado em '{modelPath}'. " +
            "Execute o script 'export_onnx.py' antes de iniciar a API.");

    return new ModelLoader(modelPath);
});

// 3. Detecção de emoção (inferência ONNX)
builder.Services.AddSingleton<EmotionDetectionService>();

// 4. Integração com o Supabase (persistência de detecções e alertas)
builder.Services.AddScoped<SupabaseService>();

// =====================================================================
// PIPELINE HTTP
// =====================================================================

var app = builder.Build();

if (app.Environment.IsDevelopment())
    app.MapOpenApi();

app.UseCors("AllowAll");
app.UseHttpsRedirection();
app.UseAuthorization();
app.MapControllers();

app.Run();
