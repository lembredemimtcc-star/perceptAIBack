using Microsoft.ML.OnnxRuntime;

namespace PerceptAI.API.ML
{
    public class ModelLoader
    {
        public InferenceSession Session { get; }

        public ModelLoader(string modelPath)
        {
            Session = new InferenceSession(modelPath);
        }
    }
}
