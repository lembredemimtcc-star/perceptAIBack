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

        // Mapeamento do modelo fisiológico (2 classes): 0: acordado, 1: dormindo
        private readonly string[] _physioClasses = { "acordado", "dormindo" };

        // Mapeamento do modelo emocional (6 classes): 0: dor, 1: enjoo, 2: medo, 3: sono, 4: tristeza, 5: neutro
        private readonly string[] _emotionClasses = { "dor", "enjoo", "medo", "sono", "tristeza", "neutro" };

        // Mapeamento legado (8 classes) para compatibilidade
        private readonly string[] _legacyEmotions = { "dor", "enjoo", "medo", "sono", "tristeza", "neutro", "acordado", "dormindo" };

        public EmotionDetectionService(ModelLoader modelLoader)
        {
            _modelLoader = modelLoader;
        }

        /// <summary>
        /// Obtém as probabilidades normalizadas (Softmax) a partir de um modelo ONNX específico.
        /// </summary>
        private float[] GetProbabilities(float[] normalizedData, InferenceSession? session)
        {
            if (session == null)
            {
                throw new InvalidOperationException("O modelo ONNX não foi carregado na memória do servidor.");
            }

            var inputTensor = new DenseTensor<float>(normalizedData, new[] { 1, 3, 224, 224 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };

            using var results = session.Run(inputs);

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
        /// Analisa os dois estados do paciente usando os 2 modelos separados:
        /// 1. Estado Fisiológico (modelo fisiológico): "dormindo" ou "acordado".
        ///    Se dormindo, EstadoEmocional é null (não precisa de emoção).
        /// 2. Se acordado: EstadoEmocional (modelo emocional) identifica a expressão com confiança >= 30%;
        ///    caso nenhuma emoção passe o limiar, retorna "neutro".
        /// </summary>
        public DualStateResult AnalyzeDualState(float[] normalizedData)
        {
            // Tenta usar os 2 modelos novos primeiro
            if (_modelLoader.PhysioSession != null && _modelLoader.EmotionSession != null)
            {
                return AnalyzeDualStateWithTwoModels(normalizedData);
            }
            // Fallback para modelo legado (único) se os novos não estiverem disponíveis
            else if (_modelLoader.PhysioSession == null && _modelLoader.EmotionSession == null)
            {
                Console.WriteLine("[EmotionDetectionService] Usando modelo legado (único) - modelos novos não carregados");
                return AnalyzeDualStateLegacy(normalizedData);
            }
            else
            {
                throw new InvalidOperationException("Configuração inválida de modelos: um dos dois modelos não foi carregado.");
            }
        }

        /// <summary>
        /// Analisa usando os 2 modelos separados (fisiológico + emocional).
        /// </summary>
        private DualStateResult AnalyzeDualStateWithTwoModels(float[] normalizedData)
        {
            // Passo 1: Modelo Fisiológico - determina se está acordado ou dormindo
            var physioProbs = GetProbabilities(normalizedData, _modelLoader.PhysioSession);
            float pAcordado = physioProbs[0];  // índice 0
            float pDormindo = physioProbs[1];  // índice 1

            // Critério: está dormindo se a probabilidade de dormindo for maior
            bool isDormindo = pDormindo > pAcordado;

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
                // Passo 2: Modelo Emocional - determina a emoção (só se estiver acordado)
                var emotionProbs = GetProbabilities(normalizedData, _modelLoader.EmotionSession);

                // Identifica a melhor emoção
                int bestEmotionIndex = 5; // neutro padrão
                float maxEmotionConf = -1.0f;
                for (int i = 0; i < emotionProbs.Length; i++)
                {
                    if (emotionProbs[i] > maxEmotionConf)
                    {
                        maxEmotionConf = emotionProbs[i];
                        bestEmotionIndex = i;
                    }
                }

                // Aplica threshold mínimo
                string emocaoDetectada;
                if (maxEmotionConf >= MinEmotionThreshold)
                {
                    emocaoDetectada = _emotionClasses[bestEmotionIndex];
                }
                else
                {
                    // Confiança insuficiente → reporta neutro (index 5)
                    bestEmotionIndex = 5;
                    emocaoDetectada = "neutro";
                    maxEmotionConf = emotionProbs[5];
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

        /// <summary>
        /// Analisa usando o modelo legado (único com 8 classes).
        /// Mantido para compatibilidade durante transição.
        /// </summary>
        private DualStateResult AnalyzeDualStateLegacy(float[] normalizedData)
        {
            // Usa a session legada (que está em PhysioSession por causa do construtor)
            var session = _modelLoader.PhysioSession;
            if (session == null)
            {
                throw new InvalidOperationException("Modelo legado não carregado.");
            }

            var probabilities = GetProbabilities(normalizedData, session);

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
                    EstadoEmocional = null,
                    ConfiancaEmocional = null,
                    PrimaryEmotion = "dormindo",
                    PrimaryConfidence = pDormindo
                };
            }
            else
            {
                string emocaoDetectada;
                if (maxEmotionConf >= MinEmotionThreshold)
                {
                    emocaoDetectada = _legacyEmotions[bestEmotionIndex];
                }
                else
                {
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
            // Se estiver usando os 2 modelos novos, retorna ambos
            if (_modelLoader.PhysioSession != null && _modelLoader.EmotionSession != null)
            {
                var physioProbs = GetProbabilities(normalizedData, _modelLoader.PhysioSession);
                var emotionProbs = GetProbabilities(normalizedData, _modelLoader.EmotionSession);

                var dict = new Dictionary<string, float>();
                for (int i = 0; i < _physioClasses.Length; i++)
                    dict[_physioClasses[i]] = physioProbs[i];
                for (int i = 0; i < _emotionClasses.Length; i++)
                    dict[_emotionClasses[i]] = emotionProbs[i];
                return dict;
            }
            // Fallback para modelo legado
            else if (_modelLoader.PhysioSession != null)
            {
                float[] probs = GetProbabilities(normalizedData, _modelLoader.PhysioSession);
                var dict = new Dictionary<string, float>();
                for (int i = 0; i < _legacyEmotions.Length && i < probs.Length; i++)
                    dict[_legacyEmotions[i]] = probs[i];
                return dict;
            }

            return new Dictionary<string, float>();
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
