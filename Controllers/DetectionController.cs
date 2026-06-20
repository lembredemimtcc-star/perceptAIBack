// DetectionController with async Supabase integration
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
            _threshold = configuration.GetValue<double>("DetectionSettings:Threshold", 0.75);
        }

        [HttpPost("detect")]
        public async Task<ActionResult<DetectionResponse>> Detect([FromBody] DetectionRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
                return BadRequest(new { message = "Corpo de requisição inválido ou imagem vazia." });

            try
            {
                float[] normalizedData = _preprocessingService.PreprocessBase64Image(request.Image);
                var (emotion, confidence) = _detectionService.DetectEmotion(normalizedData);
                string finalEmotion = confidence >= _threshold ? emotion : "neutro";
                var detectedAt = DateTime.UtcNow;

                if (finalEmotion != "neutro" && !string.IsNullOrWhiteSpace(request.PatientId))
                {
                    _ = _supabaseService.SaveDetectionAsync(request.PatientId, finalEmotion, confidence, detectedAt);
                }

                return Ok(new DetectionResponse
                {
                    Emotion = finalEmotion,
                    Confidence = confidence,
                    Timestamp = detectedAt,
                });
            }
            catch (FormatException ex)
            {
                return BadRequest(new { message = "Formato base64 inválido.", error = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { message = "Erro interno ao processar detecção.", error = ex.Message });
            }
        }
    }
}
