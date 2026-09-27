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
            // Validação básica da imagem
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
            {
                return BadRequest(new { message = "Corpo de requisição inválido ou imagem vazia." });
            }

            // Exige PatientId (fluxo legado) OU InternacaoId (fluxo web)
            bool temPatient    = !string.IsNullOrWhiteSpace(request.PatientId);
            bool temInternacao = !string.IsNullOrWhiteSpace(request.InternacaoId);

            if (!temPatient && !temInternacao)
            {
                return BadRequest(new { message = "PatientId ou InternacaoId é obrigatório." });
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

                // Persiste no Supabase dependendo do fluxo:
                if (temInternacao)
                {
                    // Fluxo web: salva em expressoes_faciais
                    await _supabaseService.SaveExpressaoAsync(
                        request.InternacaoId!,
                        emotion,
                        confidence);
                }
                else if (temPatient && confidence >= _threshold)
                {
                    // Fluxo legado do app mobile: salva em detections
                    await _supabaseService.SaveDetectionAsync(request.PatientId, emotion, confidence, response.Timestamp);
                }

                return Ok(response);
            }
            catch (FormatException)
            {
                return BadRequest(new { message = "Formato Base64 da imagem é inválido." });
            }
            catch (Exception ex)
            {
                // Retorna o detalhe da exceção para diagnóstico preciso
                return StatusCode(500, new { message = "Erro interno ao processar a detecção.", error = ex.Message, detail = ex.ToString() });
            }
        }

        /// <summary>
        /// Endpoint de diagnóstico: recebe a mesma imagem e retorna a confiança de CADA emoção.
        /// Útil para descobrir se o modelo tem viés ou se a ordem das classes está errada.
        /// POST /api/detection/debug
        /// </summary>
        [HttpPost("debug")]
        public IActionResult Debug([FromBody] DetectionRequest request)
        {
            if (request == null || string.IsNullOrWhiteSpace(request.Image))
                return BadRequest(new { message = "Imagem vazia." });

            try
            {
                float[] normalized = _preprocessingService.PreprocessBase64Image(request.Image);
                var allProbs = _detectionService.GetAllProbabilities(normalized);

                // Ordena por confiança decrescente para facilitar leitura
                var sorted = allProbs
                    .OrderByDescending(kv => kv.Value)
                    .Select(kv => new { emocao = kv.Key, confianca = Math.Round(kv.Value * 100, 2) });

                return Ok(new
                {
                    vencedor = sorted.First().emocao,
                    ranking = sorted
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { message = ex.Message });
            }
        }
    }
}
