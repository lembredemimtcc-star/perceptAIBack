using Microsoft.ML.OnnxRuntime;

namespace PerceptAI.API.ML
{
    /// <summary>
    /// Carrega os dois modelos ONNX separados:
    /// - Modelo Fisiológico (acordado/dormindo)
    /// - Modelo Emocional (6 emoções)
    /// </summary>
    public class ModelLoader
    {
        public InferenceSession? PhysioSession { get; }
        public InferenceSession? EmotionSession { get; }

        /// <summary>
        /// Sessão ativa para compatibilidade legada (health checks, etc.)
        /// </summary>
        public InferenceSession? Session => PhysioSession ?? EmotionSession;

        public ModelLoader(string? physioModelPath, string? emotionModelPath)
        {
            // Carrega modelo fisiológico (acordado/dormindo)
            if (!string.IsNullOrWhiteSpace(physioModelPath) && System.IO.File.Exists(physioModelPath))
            {
                PhysioSession = new InferenceSession(physioModelPath);
            }

            // Carrega modelo emocional (6 emoções)
            if (!string.IsNullOrWhiteSpace(emotionModelPath) && System.IO.File.Exists(emotionModelPath))
            {
                EmotionSession = new InferenceSession(emotionModelPath);
            }
        }

        /// <summary>
        /// Construtor legado para compatibilidade (carrega apenas modelo único)
        /// </summary>
        public ModelLoader(string? modelPath) : this(modelPath, null)
        {
            // Mantém compatibilidade com código antigo
            // Se usar este construtor, EmotionSession será null
        }
    }
}
