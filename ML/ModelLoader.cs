using Microsoft.ML.OnnxRuntime;

namespace PerceptAI.API.ML
{
    public class ModelLoader
    {
        public InferenceSession? Session { get; }

        public ModelLoader(string? modelPath)
        {
            if (!string.IsNullOrWhiteSpace(modelPath) && System.IO.File.Exists(modelPath))
            {
                Session = new InferenceSession(modelPath);
            }
        }
    }
}
