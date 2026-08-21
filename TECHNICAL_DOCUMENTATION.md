# Documentação Técnica: PerceptAI.API

## 1. Visão Geral do Back-end

O **PerceptAI.API** é a espinha dorsal do sistema PerceptAI. Ele serve como o motor de inferência que processa imagens capturadas de pacientes em UTI, detecta expressões faciais ou estados fisiológicos usando um modelo de Inteligência Artificial e registra essas ocorrências no banco de dados para posterior análise e alertas.

### Stack Técnico Completo
- **Framework:** ASP.NET Core (.NET 10)
- **Inteligência Artificial:** ONNX Runtime (`Microsoft.ML.OnnxRuntime`)
- **Manipulação de Imagens:** SixLabors.ImageSharp
- **Banco de Dados/BaaS:** Supabase (`supabase-csharp`)
- **Arquitetura de Rede:** REST API (JSON / Base64 payload)

### Fluxo Completo da Requisição
1. **App Mobile (Frontend)** captura uma foto do paciente via câmera.
2. A foto é comprimida e convertida para uma string **Base64**.
3. O app envia um `POST` para `/api/detection/detect` contendo a string Base64 e os dados do paciente.
4. A **API (ASP.NET Core)** recebe o payload e converte o Base64 de volta para um buffer de bytes.
5. O `ImagePreprocessingService` redimensiona, normaliza e converte a imagem em um Tensor 3D (`NCHW`).
6. O `EmotionDetectionService` recebe o Tensor, invoca o **ONNX Runtime** e recupera as probabilidades.
7. Se a probabilidade da classe predominante for $\ge 0.75$, a API aciona o `SupabaseService`.
8. O registro é persistido na nuvem (**Supabase**).
9. A API retorna um JSON com o resultado (classe detectada e nível de confiança) ou retorna que não houve detecção confiável.

---

## 2. Pipeline de Detecção (Detalhado)

### Recebimento da Imagem Base64
O Controller `DetectionController` recebe um DTO contendo a imagem em Base64:
```csharp
public class DetectionRequest
{
    public string ImageBase64 { get; set; }
    public string PacienteId { get; set; }
}
```

### Pré-processamento (`ImagePreprocessingService`)
Para o modelo analisar a imagem corretamente, ela precisa passar pelas mesmas transformações aplicadas durante o treinamento:
1. **Resize:** Redimensionamento forçado para **224x224 pixels**.
2. **Normalização (ImageNet):** 
   - Média (Means): `[0.485, 0.456, 0.406]`
   - Desvio Padrão (Stds): `[0.229, 0.224, 0.225]`
3. **Tensor NCHW:** Os pixels são rearranjados do formato Interleaved (RGB RGB RGB) para Planar (RRR GGG BBB) e convertidos num Array 1D do tipo `float`, compatível com a estrutura de tensores do ONNX `(BatchSize: 1, Channels: 3, Height: 224, Width: 224)`.

### Inferência com ONNX Runtime (`EmotionDetectionService`)
Carregamento e inferência eficientes:
```csharp
var inputTensor = new DenseTensor<float>(imageFloatArray, new[] { 1, 3, 224, 224 });
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("images", inputTensor)
};
using var results = _inferenceSession.Run(inputs);
var outputTensor = results.First().AsTensor<float>();
```
Um `Softmax` é aplicado à saída do tensor (logits) para obter as probabilidades percentuais.

### Persistência (`SupabaseService`)
Caso `confidence >= 0.75`, o registro é montado e inserido de forma assíncrona usando o SDK do Supabase C#:
```csharp
var detectionRecord = new DetectionModel
{
    PacienteId = pacienteId,
    Emocao = detectedClass,
    Confianca = confidence,
    DataHora = DateTime.UtcNow
};
await _supabaseClient.From<DetectionModel>().Insert(detectionRecord);
```

---

## 3. Modelo de IA — Treinamento

O modelo subjacente foi treinado usando **PyTorch** e posteriormente exportado para o formato padrão da indústria, o **ONNX**.

- **Arquitetura:** MobileNetV2 (Feature Extractor) + Fully Connected Layer personalizado para 8 classes. Técnica de Transfer Learning (pesos pré-treinados na ImageNet).
- **Dataset:** 1.836 imagens, divididas em 8 classes.
- **Treinamento:**
  - 5 Épocas (Otimizador Adam, Learning Rate de `1e-4`).
  - Função de perda: Cross-Entropy Loss.
  - Data Augmentation: Rotações ($\pm 15^\circ$), espelhamentos horizontais, variações de brilho e contraste.
- **Exportação:** PyTorch `torch.onnx.export` com dummy input de shape `(1, 3, 224, 224)`.

### Métricas Finais
- **Acurácia Geral:** 95,38%
- **Precisão Macro:** 92,26%
- **Recall Macro:** 91,10%
- **F1-score Macro:** 91,05%

*(Classes com melhor performance geral tendem a ser "acordado" e "neutro", seguidas por expressões fortes como "dor" e "medo")*

---

## 4. Limitações Técnicas e de Carga

1. **Tamanho Máximo de Imagem:** Como recebemos via Base64 (que aumenta em 33% o tamanho original), o limite de tamanho do payload da requisição no ASP.NET precisa ser respeitado. Recomenda-se enviar imagens muito comprimidas (ex: 50-100KB JPEG no lado do React Native).
2. **Tempo de Processamento:** Aproximadamente **30-100ms** na API (excluindo latência de rede), variando conforme a CPU do servidor.
3. **Requisições Simultâneas:** O ONNX Runtime pode consumir muita CPU e RAM em concorrência pesada. Limita-se pela thread pool do .NET e hardware do servidor. A sessão de inferência é Singleton para não estourar a memória.
4. **Armazenamento no Supabase:** Limites do plano "Free" do Supabase (500MB database, 2GB bandwidth). Cuidado com muitos registros ao longo dos meses.
5. **Threshold (0.75):** Utilizado para evitar *falsos positivos*. Abaixo de 75% de certeza, o modelo não salva o registro para não causar pânico falso na equipe de enfermagem.
6. **Tamanho do ONNX:** O arquivo `model.onnx` original pesa em torno de **13MB a 15MB** (FP32).
7. **Memória RAM:** A API consome em repouso ~150MB. O carregamento do modelo ONNX exige cerca de 50MB-100MB adicionais.
8. **Processamento:** Funciona em CPU padrão (`CPUExecutionProvider` do ONNX). Não requer GPU dedicada para o tempo de resposta estipulado.

---

## 5. Regras e Restrições (Crítico)

- **Ordem de Mapeamento (NUNCA ALTERAR):**
  O logit retornado pelo modelo mapeia rigidamente os índices para:
  `0: dor, 1: enjoo, 2: medo, 3: sono, 4: tristeza, 5: neutro, 6: acordado, 7: dormindo`
- **Dimensão e Formato:**
  Obrigatoriamente 224x224, normalização ImageNet, layout NCHW (batch dimension no início).
- **Perfil de Execução:** Sempre rode em **"http"**. O IIS Express gera problemas no mapeamento de IP local limitando a rede.
- **Segurança de Credenciais:** As variáveis do Supabase (Url e Anon/Role Key) vivem no `appsettings.json`. Elas **nunca** devem ser commitadas. 

---

## 6. Problemas Conhecidos e Status

- ✅ **Mapeamento de classes:** CORRIGIDO (estava invertido, agora está perfeitamente alinhado com o Python).
- ✅ **Visual Studio IIS Express:** CORRIGIDO (o `launchSettings.json` agora prioriza a execução Kestrel direta "http").
- ✅ **useHttpsRedirection:** COMENTADO (estava redirecionando para HTTPS inexistente e causando erro 404 no app React Native).
- ⚠️ **PENDENTE:** Erro 404/Network Error persistente ao chamar API do celular físico via IP local (mesmo com curl funcionando no PC). Necessita de configuração do firewall/bind local no ambiente do usuário.
- ⚠️ **PENDENTE:** Chamadas de diagnóstico `Console.WriteLine` estão sujando o console (necessário substituir por logger oficial no Program.cs/Services).
- ⚠️ **PENDENTE:** Modelo ONNX (`ML/model.onnx`) não está no repositório. Tem que ser gerado localmente pelo pipeline de treinamento Python e movido.

---

## 7. Próximos Passos (Roadmap Técnico)

- [ ] Investigar regras do Firewall do Windows / Bind do Kestrel que causam bloqueio 404 vindo do aplicativo físico.
- [ ] Remover `Console.WriteLine` de depuração temporária e injetar `ILogger<T>`.
- [ ] Configurar CI/CD para compilação via GitHub Actions e deploy em Cloud (ex: Azure, Render, AWS).
- [ ] Adicionar projeto de Testes Unitários (xUnit) focando primeiramente na exatidão do `ImagePreprocessingService`.
- [ ] Implementar Logging estruturado (ex: Serilog) persistindo erros no Supabase ou sistema de log centralizado.
- [ ] Rate Limiting nativo do ASP.NET Core para prevenir abusos da API.
- [ ] Expandir o modelo (retreinar) para adicionar detecção de convulsões baseadas em espasmos (possivelmente mudando para modelo focado em vídeo/frames sequenciais).
- [ ] Otimização ONNX: Aplicar Quantization (FP32 -> INT8) para reduzir tamanho de 13MB para ~3MB e dobrar velocidade de inferência na CPU.
- [ ] Cache na borda ou In-Memory das previsões para imagens exatamente iguais (hash-based).

---

## 8. Como Executar Localmente

### Pré-requisitos
1. .NET 10 (ou compatível).
2. Arquivo `ML/model.onnx` válido colocado dentro da pasta `ML/`.

### Configuração
Crie um `appsettings.Development.json` ou edite o `appsettings.json` na raiz:
```json
{
  "Logging": { "LogLevel": { "Default": "Information" } },
  "AllowedHosts": "*",
  "Supabase": {
    "Url": "https://seu-id.supabase.co",
    "Key": "sua-anon-key"
  }
}
```

### Rodar via CLI (ou use o Visual Studio - F5)
```bash
dotnet restore
dotnet build
dotnet run --launch-profile http
```

### Teste Rápido (Curl)
```bash
curl -X POST http://localhost:5198/api/detection/detect \
-H "Content-Type: application/json" \
-d "{\"pacienteId\":\"paciente123\", \"imageBase64\":\"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==\"}"
```

---

## 9. Estrutura de Pastas Detalhada

```text
PerceptAI.API/
├── Controllers/
│   └── DetectionController.cs       # Roteador POST /api/detection/detect
├── Models/
│   ├── DetectionRequest.cs          # Payload de entrada (Base64 + PacienteId)
│   └── DetectionModel.cs            # Modelo mapeado para a tabela do Supabase
├── Services/
│   ├── EmotionDetectionService.cs   # Coração da IA (InferenceSession Singleton)
│   ├── ImagePreprocessingService.cs # Magia dos tensores e ImageSharp
│   └── SupabaseService.cs           # Comunicação CRUD via Supabase-CSharp
├── ML/
│   └── model.onnx                   # Rede Neural (Não versionado!)
├── Properties/
│   └── launchSettings.json          # Regras das portas (5198) e Kestrel
├── Program.cs                       # DI Container, Middlewares, CORS e Bootstrapping
└── appsettings.json                 # Env vars
```
