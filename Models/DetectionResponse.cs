using System;

namespace PerceptAI.API.Models
{
    public class DetectionResponse
    {
        public string Emotion { get; set; } = string.Empty;
        public float Confidence { get; set; }
        public DateTime Timestamp { get; set; }
    }
}
