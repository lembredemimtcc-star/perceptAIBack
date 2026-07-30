using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using PerceptAI.API.Models;
using PerceptAI.API.Services;

namespace PerceptAI.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class DetectionController : ControllerBase
    {
        private readonly ImagePreprocessingService _preprocessingService;
        private readonly EmotionDetectionService _detectionService;
        private readonly SupabaseService _supabaseService;
        private readonly double _threshold;

        public DetectionController(
            ImagePreprocessingService preprocessingService,
            EmotionDetectionService detectionService,
            SupabaseService supabaseService,
            IConfiguration configuration)
        {
            _preprocessingService = preprocessingService;
            _detectionService = detectionService;
            _supabaseService = supabaseService;
            // Carrega o threshold padrão (75%) do appsettings.json
            _threshold = configuration.GetValue<double>("DetectionSettings:Threshold", 0.75);
        }

        /// <summary>
        /// Recebe uma imagem em Base64 e o ID do paciente, executa a inferência ONNX
        /// e persiste a detecção no Supabase (caso a confiança supere o threshold).
        /// </summary>
        [HttpPost("detect")]
        public async Task<IActionResult> Detect([FromBody] DetectionRequest request)
        {
            // Validação básica
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
            {
                return BadRequest(new { message = "Corpo de requisição inválido ou imagem vazia." });
            }

            try
            {
                // Pré-processamento da imagem Base64
                float[] normalized = _preprocessingService.PreprocessBase64Image(request.Image);

                // Inferência do modelo
                var (emotion, confidence) = _detectionService.DetectEmotion(normalized);

                // Monta a resposta
                var response = new DetectionResponse
                {
                    Emotion = emotion,
                    Confidence = confidence,
                    Timestamp = DateTime.UtcNow
                };

                // Persiste no Supabase se a confiança atingir o limiar
                if (confidence >= _threshold)
                {
                    await _supabaseService.SaveDetectionAsync(request.PatientId, emotion, confidence, response.Timestamp);
                }

                return Ok(response);
            }
            catch (FormatException)
            {
                return BadRequest(new { message = "Formato Base64 da imagem é inválido." });
            }
            catch (Exception)
            {
                // Erro interno inesperado
                return StatusCode(500, new { message = "Erro interno ao processar a detecção." });
            }
        }
    }
}
