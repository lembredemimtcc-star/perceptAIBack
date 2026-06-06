using System;
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
        private readonly double _threshold;

        public DetectionController(
            ImagePreprocessingService preprocessingService,
            EmotionDetectionService detectionService,
            IConfiguration configuration)
        {
            _preprocessingService = preprocessingService;
            _detectionService = detectionService;
            
            // Carrega o threshold padrão (75%) do appsettings.json
            _threshold = configuration.GetValue<double>("DetectionSettings:Threshold", 0.75);
        }

        [HttpPost("detect")]
        public ActionResult<DetectionResponse> Detect([FromBody] DetectionRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
            {
                return BadRequest(new { message = "Corpo de requisição inválido ou imagem vazia." });
            }

            try
            {
                // 1. Pré-processa a imagem (Base64 -> ImageSharp -> Redimensionamento -> Normalização NCHW)
                float[] normalizedData = _preprocessingService.PreprocessBase64Image(request.Image);

                // 2. Executa a inferência com o modelo ONNX
                var (emotion, confidence) = _detectionService.DetectEmotion(normalizedData);

                // 3. Verifica o limite (threshold) configurado no appsettings.json
                // Se a confiança for menor que o limiar, retorna estado "neutro"
                string finalEmotion = emotion;
                if (confidence < _threshold)
                {
                    finalEmotion = "neutro";
                }

                // 4. Constrói a resposta
                var response = new DetectionResponse
                {
                    Emotion = finalEmotion,
                    Confidence = confidence,
                    Timestamp = DateTime.UtcNow
                };

                return Ok(response);
            }
            catch (FormatException ex)
            {
                return BadRequest(new { message = "Formato base64 de imagem inválido.", error = ex.Message });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { message = "Erro interno ao processar detecção de emoção.", error = ex.Message });
            }
        }
    }
}
