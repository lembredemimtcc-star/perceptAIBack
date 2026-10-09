using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using PerceptAI.API.ML;
using PerceptAI.API.Services;
using PerceptAI.API.Settings;
using Scalar.AspNetCore;
// Debug statements removed
var builder = WebApplication.CreateBuilder(args);

// Read PORT from environment (required for Render, Railway, etc.)
var port = Environment.GetEnvironmentVariable("PORT");
if (!string.IsNullOrWhiteSpace(port))
{
    builder.WebHost.UseUrls($"http://0.0.0.0:{port}");
}

// Carrega .env local e mapeia para SupabaseSettings
var envPath = Path.Combine(Directory.GetCurrentDirectory(), ".env");
if (File.Exists(envPath))
{
    foreach (var line in File.ReadAllLines(envPath))
    {
        if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith('#')) continue;
        var idx = line.IndexOf('=');
        if (idx <= 0) continue;
        var key   = line[..idx].Trim().Trim('"');
        var value = line[(idx + 1)..].Trim().Trim('"');
        builder.Configuration[key] = value;
    }
    // URL do Supabase (compartilhada com o front)
    var supabaseUrl = builder.Configuration["VITE_SUPABASE_URL"]
                   ?? builder.Configuration["SUPABASE_URL"]
                   ?? "";
    // Service Role Key: variável dedicada do backend (prefira esta)
    // ou fallback para VITE_SUPABASE_ANON_KEY se não houver outra
    var serviceRoleKey = builder.Configuration["SUPABASE_SERVICE_ROLE_KEY"]
                      ?? builder.Configuration["VITE_SUPABASE_SERVICE_ROLE_KEY"]
                      ?? builder.Configuration["VITE_SUPABASE_ANON_KEY"]
                      ?? "";

    if (!string.IsNullOrWhiteSpace(supabaseUrl))
        builder.Configuration["SupabaseSettings:Url"] = supabaseUrl;
    if (!string.IsNullOrWhiteSpace(serviceRoleKey))
        builder.Configuration["SupabaseSettings:ServiceRoleKey"] = serviceRoleKey;
}

// ------------------------------------------------------------------
// CONFIGURAÇÕES
// ------------------------------------------------------------------
// Bind Supabase settings
builder.Services.Configure<SupabaseSettings>(builder.Configuration.GetSection(SupabaseSettings.SectionName));

// ------------------------------------------------------------------
// CORS
// ------------------------------------------------------------------
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

// ------------------------------------------------------------------
// OpenAPI / Swagger (Scalar)
// ------------------------------------------------------------------
builder.Services.AddOpenApi();
// builder.Services.AddScalarApiReference(); // Removed: method not available on IServiceCollection

// ------------------------------------------------------------------
// Serviços de negócio
// ------------------------------------------------------------------
builder.Services.AddControllers();
builder.Services.AddSingleton<ImagePreprocessingService>();
builder.Services.AddSingleton(sp =>
{
    // Caminhos para os 2 modelos separados
    var physioModelPath = Path.Combine(AppContext.BaseDirectory, "ML", "perceptai_physio.onnx");
    var emotionModelPath = Path.Combine(AppContext.BaseDirectory, "ML", "perceptai_emotion.onnx");

    // Fallback para diretório atual
    if (!File.Exists(physioModelPath))
        physioModelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "perceptai_physio.onnx");
    if (!File.Exists(emotionModelPath))
        emotionModelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "perceptai_emotion.onnx");

    // Verifica se os novos modelos existem
    bool physioExists = File.Exists(physioModelPath);
    bool emotionExists = File.Exists(emotionModelPath);

    // Se os novos modelos existem, carrega os 2
    if (physioExists && emotionExists)
    {
        try
        {
            Console.WriteLine($"[ModelLoader] Carregando modelo fisiológico: {physioModelPath}");
            Console.WriteLine($"[ModelLoader] Carregando modelo emocional: {emotionModelPath}");
            return new ModelLoader(physioModelPath, emotionModelPath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AVISO CRÍTICO] Falha ao carregar modelos ONNX: {ex.Message}");
        }
    }
    else
    {
        // Fallback para modelo legado (único) se os novos não existirem
        var legacyModelPath = Path.Combine(AppContext.BaseDirectory, "ML", "model.onnx");
        if (!File.Exists(legacyModelPath))
            legacyModelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "model.onnx");

        if (File.Exists(legacyModelPath))
        {
            Console.WriteLine($"[ModelLoader] Carregando modelo legado (único): {legacyModelPath}");
            try
            {
                return new ModelLoader(legacyModelPath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[AVISO CRÍTICO] Falha ao carregar modelo legado: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine($"[AVISO CRÍTICO] Nenhum modelo ONNX encontrado:");
            Console.WriteLine($"  - Fisiológico: {physioModelPath} ({(physioExists ? "EXISTS" : "NOT FOUND")})");
            Console.WriteLine($"  - Emocional: {emotionModelPath} ({(emotionExists ? "EXISTS" : "NOT FOUND")})");
            Console.WriteLine($"  - Legado: {legacyModelPath} (NOT FOUND)");
        }
    }

    // Retorna um loader nulo controlado se nenhum modelo for encontrado
    return new ModelLoader(null, null);
});
builder.Services.AddSingleton<EmotionDetectionService>();
builder.Services.AddScoped<SupabaseService>();

// ------------------------------------------------------------------
// HttpClient para Supabase (pré-configurado)
// ------------------------------------------------------------------
builder.Services.AddHttpClient("supabase", (sp, client) =>
{
    var settings = sp.GetRequiredService<IOptions<SupabaseSettings>>().Value;
    if (string.IsNullOrWhiteSpace(settings.Url))
        throw new InvalidOperationException("SupabaseSettings:Url não configurado.");
    if (string.IsNullOrWhiteSpace(settings.ServiceRoleKey))
        throw new InvalidOperationException("SupabaseSettings:ServiceRoleKey não configurado.");
    client.BaseAddress = new Uri($"{settings.Url.TrimEnd('/')}/rest/v1/");
    client.DefaultRequestHeaders.Add("apikey", settings.ServiceRoleKey);
    client.DefaultRequestHeaders.Add("Authorization", $"Bearer {settings.ServiceRoleKey}");
    client.DefaultRequestHeaders.Add("Accept", "application/json");
});
// Debug statement removed
var app = builder.Build();

// Debug statement removed

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.MapScalarApiReference();
    app.MapGet("/swagger", async ctx =>
    {
        ctx.Response.Redirect("/scalar/v1");
        await Task.CompletedTask;
    });
}

app.UseCors("AllowAll");
// app.UseHttpsRedirection(); // Disabled for local development
app.UseAuthorization();
        // Debug endpoint – runs a single local image through the full pipeline
        app.MapGet("/debug/test", async (Microsoft.AspNetCore.Http.HttpContext ctx) =>
        {
            var imgPath = Path.Combine(Directory.GetCurrentDirectory(), "test-images", "example.jpg");
            if (!File.Exists(imgPath))
                return Results.NotFound("Test image not found");

            var imgBytes = await File.ReadAllBytesAsync(imgPath);
            var base64 = Convert.ToBase64String(imgBytes);

            var preprocessor = ctx.RequestServices.GetRequiredService<ImagePreprocessingService>();
            var normalized = preprocessor.PreprocessBase64Image(base64);

            var detector = ctx.RequestServices.GetRequiredService<EmotionDetectionService>();
            var (emotion, confidence) = detector.DetectEmotion(normalized);

            return Results.Json(new { emotion, confidence });
        });

// Debug statement removed
app.MapControllers();
app.Run();

