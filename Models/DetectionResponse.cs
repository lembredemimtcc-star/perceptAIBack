using System;

namespace PerceptAI.API.Models
{
    public class DetectionResponse
    {
        public string Emotion { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public string EstadoFisiologico { get; set; } = "acordado"; // "acordado" ou "dormindo"
        public float ConfiancaFisiologica { get; set; }
        public string? EstadoEmocional { get; set; } // "dor", "enjoo", "medo", "tristeza", "neutro", "sono" (null se dormindo)
        public float? ConfiancaEmocional { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
