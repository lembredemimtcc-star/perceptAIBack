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
        private readonly string[] _emotions = { "dor", "enjoo", "medo", "sono", "tristeza", "neutro", "acordado", "dormindo" };

        public EmotionDetectionService(ModelLoader modelLoader)
        {
            _modelLoader = modelLoader;
        }

        public (string emotion, float confidence) DetectEmotion(float[] normalizedData)
        {
            // 1. Cria o tensor denso no formato [1, 3, 224, 224]
            var inputTensor = new DenseTensor<float>(normalizedData, new[] { 1, 3, 224, 224 });

            // 2. Prepara a entrada com o nome 'images' (conforme exportação ONNX: input_names=["images"])
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            if (_modelLoader.Session == null)
            {
                throw new InvalidOperationException("O modelo ONNX (ML/model.onnx) não foi carregado na memória do servidor.");
            }

            // 3. Executa a inferência na sessão ativa
            using var results = _modelLoader.Session.Run(inputs);

            // 4. Extrai a saída com o nome 'logits' (conforme exportação ONNX: output_names=["logits"])
            var outputValue = results.FirstOrDefault(r => r.Name == "logits");
            if (outputValue == null)
            {
                // Fallback: tenta pegar o primeiro output disponível (compatibilidade)
                outputValue = results.FirstOrDefault();
                if (outputValue == null)
                    throw new InvalidOperationException("Nenhuma saída encontrada no modelo ONNX.");
            }

            var outputTensor = outputValue.AsTensor<float>();
            float[] logits = outputTensor.ToArray();

            // Aplica Softmax para obter probabilidades reais
            float[] probabilities = Softmax(logits);

            // Se o modelo estiver dando 'neutro' (índice 5) com confiança alta mas houver outra classe forte,
            // ou se a ordem das classes do dataset do PyTorch usou ordenação alfabética de pastas:
            // 0: acordado | 1: desassordado/dormindo | 2: dor | 3: enjoo | 4: medo | 5: neutro | 6: sono | 7: tristeza
            
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

            // Se a maior classe for neutro, mas houver uma emoção clínica (dor, medo, tristeza, enjoo) com mais de 15% de confiança,
            // prioriza a emoção clínica para o monitoramento médico!
            string selectedEmotion = _emotions[maxIndex];
            if (selectedEmotion == "neutro")
            {
                int clinicalIndex = -1;
                float maxClinicalConf = 0.15f; // limiar de 15%
                for (int i = 0; i < probabilities.Length; i++)
                {
                    string name = _emotions[i];
                    if (name != "neutro" && name != "acordado" && name != "dormindo" && probabilities[i] > maxClinicalConf)
                    {
                        maxClinicalConf = probabilities[i];
                        clinicalIndex = i;
                    }
                }
                if (clinicalIndex >= 0)
                {
                    selectedEmotion = _emotions[clinicalIndex];
                    maxConfidence = probabilities[clinicalIndex];
                }
            }

            return (selectedEmotion, maxConfidence);
        }

        /// <summary>
        /// Retorna todas as probabilidades para diagnóstico (útil para depurar viés do modelo).
        /// </summary>
        public Dictionary<string, float> GetAllProbabilities(float[] normalizedData)
        {
            var inputTensor = new DenseTensor<float>(normalizedData, new[] { 1, 3, 224, 224 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor) // nome correto: "images"
            };
            using var results = _modelLoader.Session.Run(inputs);
            var outputValue = results.FirstOrDefault(r => r.Name == "logits") // nome correto: "logits"
                           ?? results.FirstOrDefault();
            if (outputValue == null) return new Dictionary<string, float>();

            float[] logits = outputValue.AsTensor<float>().ToArray();
            float[] probs = Softmax(logits);

            var dict = new Dictionary<string, float>();
            for (int i = 0; i < _emotions.Length && i < probs.Length; i++)
                dict[_emotions[i]] = probs[i];
            return dict;
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
