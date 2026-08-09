using System.ComponentModel.DataAnnotations;

namespace PerceptAI.API.Models
{
    public class DetectionRequest
    {
        [Required(ErrorMessage = "A imagem base64 é obrigatória.")]
        public string Image { get; set; } = string.Empty;

        /// <summary>
        /// ID do paciente (fluxo legado do app mobile).
        /// Opcional quando InternacaoId estiver presente.
        /// </summary>
        public string PatientId { get; set; } = string.Empty;

        /// <summary>
        /// UUID da internação no Supabase (fluxo web).
        /// Quando presente, persiste em expressoes_faciais em vez de detections.
        /// </summary>
        public string? InternacaoId { get; set; }
    }
}
