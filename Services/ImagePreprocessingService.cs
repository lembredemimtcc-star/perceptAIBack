using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PerceptAI.API.Services
{
    public class ImagePreprocessingService
    {
        public float[] PreprocessBase64Image(string base64Image)
        {
            // 1. Limpa o cabeçalho base64 caso exista (ex: data:image/jpeg;base64,...)
            if (base64Image.Contains(","))
            {
                base64Image = base64Image.Split(',')[1];
            }

            // 2. Converte para array de bytes
            byte[] imageBytes = Convert.FromBase64String(base64Image);

            // 3. Carrega a imagem usando ImageSharp
            using var image = Image.Load<Rgb24>(imageBytes);

            // 4. Redimensiona para 224x224
            image.Mutate(x => x.Resize(224, 224));

            // 5. Normaliza os pixels e prepara o layout NCHW (Channels First) esperado pelo MobileNetV2
            // Tamanho: 1 canal * 3 cores * 224 largura * 224 altura
            float[] normalizedData = new float[1 * 3 * 224 * 224];

            // Offsets dos canais R, G e B na memória linear
            int rOffset = 0;
            int gOffset = 224 * 224;
            int bOffset = 2 * 224 * 224;

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> row = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        int index = y * 224 + x;
                        Rgb24 pixel = row[x];

                        // Normalização padrão ImageNet: (Pixel / 255.0 - Média) / Desvio Padrão
                        normalizedData[rOffset + index] = ((pixel.R / 255.0f) - 0.485f) / 0.229f;
                        normalizedData[gOffset + index] = ((pixel.G / 255.0f) - 0.456f) / 0.224f;
                        normalizedData[bOffset + index] = ((pixel.B / 255.0f) - 0.406f) / 0.225f;
                    }
                }
            });

            return normalizedData;
        }
    }
}
