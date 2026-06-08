using System.ComponentModel.DataAnnotations;

namespace PerceptAI.API.Models
{
    public class DetectionRequest
    {
        [Required(ErrorMessage = "A imagem base64 é obrigatória.")]
        public string Image { get; set; } = string.Empty;

        [Required(ErrorMessage = "O patientId é obrigatório.")]
        public string PatientId { get; set; } = string.Empty;
    }
}
