using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PerceptAI.API.ML;

namespace PerceptAI.API.Services
{
    public class DualStateResult
    {
        public string EstadoFisiologico { get; set; } = "acordado"; // "acordado" ou "dormindo"
        public float ConfiancaFisiologica { get; set; }
        public string? EstadoEmocional { get; set; }                 // "dor", "enjoo", "medo", "tristeza", "neutro", "sono" (null se dormindo)
        public float? ConfiancaEmocional { get; set; }
        public string PrimaryEmotion { get; set; } = string.Empty;
        public float PrimaryConfidence { get; set; }
    }

    public class EmotionDetectionService
    {
        private readonly ModelLoader _modelLoader;
        
        // Mapeamento das 8 classes na ordem exata de saída do classificador da rede:
        // 0: dor | 1: enjoo | 2: medo | 3: sono | 4: tristeza | 5: neutro | 6: acordado | 7: dormindo
        private readonly string[] _emotions = { "dor", "enjoo", "medo", "sono", "tristeza", "neutro", "acordado", "dormindo" };

        public EmotionDetectionService(ModelLoader modelLoader)
        {
            _modelLoader = modelLoader;
        }

        /// <summary>
        /// Obtém as probabilidades normalizadas (Softmax) de todas as 8 classes a partir do tensor de entrada.
        /// </summary>
        public float[] GetProbabilities(float[] normalizedData)
        {
            var inputTensor = new DenseTensor<float>(normalizedData, new[] { 1, 3, 224, 224 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            if (_modelLoader.Session == null)
            {
                throw new InvalidOperationException("O modelo ONNX (ML/model.onnx) não foi carregado na memória do servidor.");
            }

            using var results = _modelLoader.Session.Run(inputs);

            var outputValue = results.FirstOrDefault(r => r.Name == "logits") ?? results.FirstOrDefault();
            if (outputValue == null)
            {
                throw new InvalidOperationException("Nenhuma saída encontrada no modelo ONNX.");
            }

            var outputTensor = outputValue.AsTensor<float>();
            float[] logits = outputTensor.ToArray();

            return Softmax(logits);
        }

        // Confiança mínima para reportar uma emoção específica.
        // Abaixo disso, retorna "neutro" para evitar falsos positivos com rostos ambíguos.
        private const float MinEmotionThreshold = 0.30f;

        /// <summary>
        /// Analisa os dois estados do paciente simultaneamente:
        /// 1. Estado Fisiológico: "dormindo" ou "acordado".
        ///    Se dormindo, EstadoEmocional é null (não precisa de emoção).
        /// 2. Se acordado: EstadoEmocional identifica a expressão com confiança >= 30%;
        ///    caso nenhuma emoção passe o limiar, retorna "neutro".
        /// </summary>
        public DualStateResult AnalyzeDualState(float[] normalizedData)
        {
            var probabilities = GetProbabilities(normalizedData);

            // Índices: 0: dor, 1: enjoo, 2: medo, 3: sono, 4: tristeza, 5: neutro, 6: acordado, 7: dormindo
            float pDormindo = probabilities[7];
            float pAcordado = probabilities[6];

            // Identifica a melhor emoção entre as emoções ativas (0 a 5)
            int bestEmotionIndex = 5; // neutro padrão
            float maxEmotionConf = -1.0f;
            for (int i = 0; i <= 5; i++)
            {
                if (probabilities[i] > maxEmotionConf)
                {
                    maxEmotionConf = probabilities[i];
                    bestEmotionIndex = i;
                }
            }

            // Identifica o vencedor absoluto global (entre todas as 8 classes)
            int globalMaxIndex = 0;
            float globalMaxConf = -1.0f;
            for (int i = 0; i < probabilities.Length; i++)
            {
                if (probabilities[i] > globalMaxConf)
                {
                    globalMaxConf = probabilities[i];
                    globalMaxIndex = i;
                }
            }

            // Critério: paciente está dormindo se a classe 'dormindo' for o vencedor global
            // ou se superar 'acordado' e qualquer emoção ativa com folga.
            bool isDormindo = globalMaxIndex == 7 || (pDormindo > pAcordado && pDormindo > maxEmotionConf);

            if (isDormindo)
            {
                return new DualStateResult
                {
                    EstadoFisiologico = "dormindo",
                    ConfiancaFisiologica = pDormindo,
                    EstadoEmocional = null, // Se estiver dormindo, não precisa de emoção
                    ConfiancaEmocional = null,
                    PrimaryEmotion = "dormindo",
                    PrimaryConfidence = pDormindo
                };
            }
            else
            {
                // Paciente acordado.
                // Aplica threshold mínimo: se nenhuma emoção ativa passar o limiar de confiança,
                // retorna "neutro" para evitar falsos positivos em imagens ambíguas de webcam.
                string emocaoDetectada;
                if (maxEmotionConf >= MinEmotionThreshold)
                {
                    emocaoDetectada = _emotions[bestEmotionIndex];
                }
                else
                {
                    // Confiança insuficiente → reporta neutro (index 5)
                    bestEmotionIndex = 5;
                    emocaoDetectada = "neutro";
                }

                return new DualStateResult
                {
                    EstadoFisiologico = "acordado",
                    ConfiancaFisiologica = pAcordado,
                    EstadoEmocional = emocaoDetectada,
                    ConfiancaEmocional = maxEmotionConf,
                    PrimaryEmotion = emocaoDetectada,
                    PrimaryConfidence = maxEmotionConf
                };
            }
        }

        public (string emotion, float confidence) DetectEmotion(float[] normalizedData)
        {
            var res = AnalyzeDualState(normalizedData);
            return (res.PrimaryEmotion, res.PrimaryConfidence);
        }

        /// <summary>
        /// Retorna todas as probabilidades para diagnóstico (útil para depurar viés do modelo).
        /// </summary>
        public Dictionary<string, float> GetAllProbabilities(float[] normalizedData)
        {
            if (_modelLoader.Session == null) return new Dictionary<string, float>();
            float[] probs = GetProbabilities(normalizedData);

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
