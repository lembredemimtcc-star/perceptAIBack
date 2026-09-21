# PerceptAI - Back-end API

Esta é a API responsável pelo processamento e detecção de emoções faciais do projeto **PerceptAI**. 
O PerceptAI é uma ferramenta desenvolvida para auxiliar na comunicação de pacientes em UTI que apresentam dificuldades de comunicação verbal.

A API recebe as imagens capturadas pelo aplicativo móvel, realiza o pré-processamento e utiliza um modelo treinado de inteligência artificial (via ONNX Runtime) para classificar as expressões faciais ou o estado do paciente.

## 🚀 Tecnologias Principais

- **C# / ASP.NET Core**: Framework principal da API.
- **ONNX Runtime**: Utilizado para carregar o modelo de Deep Learning treinado e realizar inferências (predições) de forma eficiente.
- **Supabase**: Banco de dados e autenticação utilizados para persistir os registros de detecção e alertas.

## 📋 Pré-requisitos

Para rodar este projeto localmente, você precisará de:

- **.NET 10 SDK** (ou a versão compatível instalada).
- **Visual Studio 2022 Community** (recomendado para desenvolvimento C#).
- **Conta no Supabase** com as credenciais do projeto.

## ⚙️ Passo a passo para rodar o projeto

1. **Abrir o projeto**: Abra a solução `PerceptAI.API` no Visual Studio.
2. **Configuração de Ambiente**:
   - Localize ou crie o arquivo `appsettings.json` na raiz do projeto (use o `appsettings.Example.json` como base).
   - Preencha as credenciais do Supabase:
     ```json
     {
       "Supabase": {
         "Url": "SUA_URL_DO_SUPABASE",
         "Key": "SUA_CHAVE_DO_SUPABASE"
       }
     }
     ```
   - *Nota: O arquivo `appsettings.json` é ignorado pelo Git por questões de segurança.*
3. **Verificação de Porta**:
   - Certifique-se de que a porta **5198** está livre. Você pode verificar rodando no terminal: `netstat -ano | findstr :5198`.
4. **Rodar a Aplicação**:
   - No Visual Studio, selecione o perfil de execução **"http"** (Nunca utilize IIS Express para testes de rede local com o app mobile).
   - Pressione **F5** para iniciar.
   - Confirme no console se aparece a mensagem: `Now listening on: http://0.0.0.0:5198`.

## ⚠️ Avisos Importantes

- Para que o aplicativo mobile (React Native/Expo) consiga se comunicar com esta API local, o computador rodando a API e o celular devem estar conectados à **mesma rede Wi-Fi**.
- Sempre verifique o seu IP local (usando `ipconfig` no terminal) e garanta que o app mobile está configurado para apontar para esse IP, pois ele muda caso o Wi-Fi reconecte.
- O modelo de IA (`ML/model.onnx` e `.data`) não está versionado neste repositório devido ao seu tamanho. Certifique-se de baixar ou treinar o modelo e colocá-lo na pasta correta antes de rodar.

## 📁 Arquivos Importantes

| Arquivo/Pasta | Descrição |
|---|---|
| `Services/ImagePreprocessingService.cs` | Pré-processa a imagem (redimensionamento, normalização, tensor NCHW) para preparar para o modelo. |
| `Services/EmotionDetectionService.cs` | Carrega o modelo ONNX e executa a inferência. |
| `Services/SupabaseService.cs` | Integração com o banco Supabase para salvar detecções e alertas. |
| `Controllers/DetectionController.cs` | Controller que expõe o endpoint POST (`/api/detection/detect`) consumido pelo app. |
| `ML/model.onnx` | Modelo treinado (arquivo não incluído no repositório por questões de tamanho/segurança). |
| `appsettings.json` | Configurações locais (credenciais) — ignorado pelo Git. |

## 🧠 Classes Detectadas pelo Modelo

| Índice | Classe | Descrição |
|---|---|---|
| 0 | dor | Expressão de dor |
| 1 | enjoo | Expressão de enjoo/náusea |
| 2 | medo | Expressão de medo |
| 3 | sono | Estado de sonolência |
| 4 | tristeza | Expressão de tristeza |
| 5 | neutro | Expressão neutra |
| 6 | acordado | Paciente acordado |
| 7 | dormindo | Paciente dormindo |
