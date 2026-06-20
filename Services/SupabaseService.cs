using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using PerceptAI.API.Settings;

namespace PerceptAI.API.Services
{
    /// <summary>
    /// Serviço responsável por persistir detecções e disparar alertas no Supabase
    /// usando a REST API PostgREST com a Service Role Key (bypass de RLS).
    /// </summary>
    public class SupabaseService
    {
        private readonly HttpClient _http;
        private readonly ILogger<SupabaseService> _logger;

        // DTO interno para inserção na tabela `detections`
        private record DetectionRow(
            string patient_id,
            string tipo_emocao,
            float confianca,
            string timestamp);

        // DTO interno para inserção na tabela `alerts`
        private record AlertRow(
            string detection_id,
            string cuidador_id,
            bool lido);

        // Resposta mínima de detecção (só precisamos do id gerado)
        private record DetectionCreated(string id);

        // Resposta mínima de patient (só precisamos do cuidador_id)
        private record PatientRow(string cuidador_id);

        public SupabaseService(
            IHttpClientFactory httpClientFactory,
            ILogger<SupabaseService> logger)
        {
            _http   = httpClientFactory.CreateClient("supabase");
            _logger = logger;
        }

        // ----------------------------------------------------------------
        // PUBLIC API
        // ----------------------------------------------------------------

        /// <summary>
        /// Salva uma detecção na tabela <c>detections</c> e dispara um alerta
        /// automático para o cuidador responsável pelo paciente.
        /// </summary>
        public async Task<bool> SaveDetectionAsync(
            string patientId,
            string emotion,
            float confidence,
            DateTime timestamp)
        {
            // 1. Inserir detecção e recuperar o id gerado pelo Supabase
            string? detectionId = await InsertDetectionAsync(patientId, emotion, confidence, timestamp);
            if (detectionId is null)
            {
                _logger.LogWarning("Detecção não pôde ser salva; alerta não será gerado.");
                return false;
            }

            // 2. Buscar o cuidador responsável pelo paciente
            string? cuidadorId = await FetchCuidadorIdAsync(patientId);
            if (cuidadorId is null)
            {
                _logger.LogWarning(
                    "Paciente {PatientId} não encontrado ou sem cuidador; alerta não será gerado.",
                    patientId);
                return false;
            }

            // 3. Criar alerta para o cuidador
            await InsertAlertAsync(detectionId, cuidadorId);

            return true;
        }

        /// <summary>
        /// Retorna o JSON bruto das últimas detecções (útil para testes).
        /// </summary>
        public async Task<string> FetchDetectionsRawAsync(int limit = 10)
        {
            try
            {
                var url = $"detections?select=*&order=timestamp.desc&limit={limit}";
                var response = await _http.GetAsync(url);
                if (!response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync();
                    _logger.LogError("Erro ao buscar detecções brutas: {Status} — {Body}", response.StatusCode, body);
                    return "[]";
                }

                return await response.Content.ReadAsStringAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exceção ao buscar detecções brutas");
                return "[]";
            }
        }

        // ----------------------------------------------------------------
        // PRIVATE HELPERS
        // ----------------------------------------------------------------

        private async Task<string?> InsertDetectionAsync(
            string patientId,
            string emotion,
            float confidence,
            DateTime timestamp)
        {
            var row = new DetectionRow(
                patient_id:  patientId,
                tipo_emocao: emotion,
                confianca:   confidence,
                timestamp:   timestamp.ToString("o")); // ISO 8601

            try
            {
                // Prefer=return=representation faz o Supabase devolver a linha inserida (com o id)
                using var request = new HttpRequestMessage(HttpMethod.Post, "detections");
                request.Headers.Add("Prefer", "return=representation");
                request.Content = JsonContent.Create(row);

                var response = await _http.SendAsync(request);

                if (!response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync();
                    _logger.LogError(
                        "Erro ao inserir detecção: {Status} — {Body}",
                        response.StatusCode, body);
                    return null;
                }

                var created = await response.Content.ReadFromJsonAsync<DetectionCreated[]>();
                return created?[0]?.id;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exceção ao inserir detecção para paciente {PatientId}", patientId);
                return null;
            }
        }

        private async Task<string?> FetchCuidadorIdAsync(string patientId)
        {
            try
            {
                // GET /patients?id=eq.{patientId}&select=cuidador_id
                var url = $"patients?id=eq.{patientId}&select=cuidador_id";
                var rows = await _http.GetFromJsonAsync<PatientRow[]>(url);
                return rows?.Length > 0 ? rows[0].cuidador_id : null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exceção ao buscar cuidador do paciente {PatientId}", patientId);
                return null;
            }
        }

        private async Task InsertAlertAsync(string detectionId, string cuidadorId)
        {
            var row = new AlertRow(
                detection_id: detectionId,
                cuidador_id:  cuidadorId,
                lido:         false);

            try
            {
                var response = await _http.PostAsJsonAsync("alerts", row);

                if (!response.IsSuccessStatusCode)
                {
                    var body = await response.Content.ReadAsStringAsync();
                    _logger.LogError(
                        "Erro ao criar alerta: {Status} — {Body}",
                        response.StatusCode, body);
                }
                else
                {
                    _logger.LogInformation(
                        "Alerta criado para cuidador {CuidadorId} (detecção {DetectionId})",
                        cuidadorId, detectionId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Exceção ao criar alerta para detecção {DetectionId}", detectionId);
            }
        }
    }
}
