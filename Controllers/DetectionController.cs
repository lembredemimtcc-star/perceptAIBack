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
            _detectionService     = detectionService;
            _supabaseService      = supabaseService;

            // Carrega o threshold padrão (75%) do appsettings.json
            _threshold = configuration.GetValue<double>("DetectionSettings:Threshold", 0.75);
        }

        /// <summary>
        /// Recebe uma imagem em Base64 e o ID do paciente, executa a inferência ONNX
        /// e persiste a detecção no Supabase (caso a confiança supere o threshold).
        /// </summary>
        [HttpPost("detect")]
        public async Task<ActionResult<DetectionResponse>> Detect([FromBody] DetectionRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
                return BadRequest(new { message = "Corpo de requisição inválido ou imagem vazia." });

            try
            {
                // 1. Pré-processa a imagem: Base64 → ImageSharp → resize → NCHW float[]
                float[] normalizedData = _preprocessingService.PreprocessBase64Image(request.Image);

                // 2. Inferência ONNX
                var (emotion, confidence) = _detectionService.DetectEmotion(normalizedData);

                // 3. Verifica o threshold — abaixo do limiar retorna "neutro"
                string finalEmotion = confidence >= _threshold ? emotion : "neutro";

                var detectedAt = DateTime.UtcNow;

                // 4. Persiste no Supabase apenas quando não é neutro
                //    (evita poluir a tabela com frames sem expressão relevante)
                if (finalEmotion != "neutro" && !string.IsNullOrWhiteSpace(request.PatientId))
                {
                    // Fire-and-forget com tratamento de erro interno no serviço:
                    // não bloqueia a resposta ao cliente em caso de falha de rede.
                    _ = _supabaseService.SaveDetectionAsync(
                        request.PatientId,
                        finalEmotion,
                        confidence,
                        detectedAt);
                }

                // 5. Retorna resultado imediatamente ao cliente (baixa latência)
                return Ok(new DetectionResponse
                {
                    Emotion    = finalEmotion,
                    Confidence = confidence,
                    Timestamp  = detectedAt,
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
