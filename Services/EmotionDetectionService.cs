using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PerceptAI.API.ML;

namespace PerceptAI.API.Services
{
    public class EmotionDetectionService
    {
        private readonly ModelLoader _modelLoader;
        
        // Mapeamento das 5 emoções na ordem exata de saída do classificador da rede
        private readonly string[] _emotions = { "medo", "enjoo", "dor", "sono", "tristeza" };

        public EmotionDetectionService(ModelLoader modelLoader)
        {
            _modelLoader = modelLoader;
        }

        public (string emotion, float confidence) DetectEmotion(float[] normalizedData)
        {
            // 1. Cria o tensor denso no formato [1, 3, 224, 224]
            var inputTensor = new DenseTensor<float>(normalizedData, new[] { 1, 3, 224, 224 });

            // 2. Prepara a entrada da sessão com o nome 'input' configurado na exportação ONNX
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 3. Executa a inferência na sessão ativa
            using var results = _modelLoader.Session.Run(inputs);

            // 4. Extrai a saída com o nome 'output'
            var outputValue = results.FirstOrDefault(r => r.Name == "output");
            if (outputValue == null)
            {
                throw new InvalidOperationException("Não foi possível encontrar a camada de saída 'output' no modelo ONNX.");
            }

            var outputTensor = outputValue.AsTensor<float>();
            float[] logits = outputTensor.ToArray();

            // 5. Aplica a função matemática Softmax para obter as probabilidades reais [0, 1]
            float[] probabilities = Softmax(logits);

            // 6. Encontra o índice da emoção com maior grau de confiança
            int maxIndex = 0;
            float maxConfidence = -1.0f;

            for (int i = 0; i < probabilities.Length; i++)
            {
                if (probabilities[i] > maxConfidence)
                {
                    maxConfidence = probabilities[i];
                    maxIndex = i;
                }
            }

            return (_emotions[maxIndex], maxConfidence);
        }

        /// <summary>
        /// Aplica a função de ativação Softmax para normalizar logits em probabilidades.
        /// </summary>
        private float[] Softmax(float[] logits)
        {
            float max = logits.Max();
            float[] exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
            float sum = exps.Sum();
            
            if (sum == 0) return logits.Select(_ => 1.0f / logits.Length).ToArray();

            return exps.Select(x => x / sum).ToArray();
        }
    }
}
