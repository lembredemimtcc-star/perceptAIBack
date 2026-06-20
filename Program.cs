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

var builder = WebApplication.CreateBuilder(args);

// Bind to all network interfaces for both HTTP and HTTPS (development only)
builder.WebHost.UseUrls("http://0.0.0.0:5198", "https://0.0.0.0:7290");

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
    var modelPath = Path.Combine(AppContext.BaseDirectory, "ML", "model.onnx");
    if (!File.Exists(modelPath))
    {
        modelPath = Path.Combine(Directory.GetCurrentDirectory(), "ML", "model.onnx");
    }
    if (!File.Exists(modelPath))
    {
        throw new FileNotFoundException($"O modelo ONNX não foi encontrado em '{modelPath}'.");
    }
    return new ModelLoader(modelPath);
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

var app = builder.Build();

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
app.MapControllers();
app.Run();
