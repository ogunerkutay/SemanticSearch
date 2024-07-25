using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace yapayzezkav1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Get the desktop folder path for the current user
            string desktopPath = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

            // Append your file name to the desktop path
            string filePath = System.IO.Path.Combine(desktopPath, "news.txt");
            string query = "Motorin satış fiyatı";

            List<string> lines = ReadLinesFromFile(filePath);
            List<string> searchResults = SemanticSearch(lines, query);

            foreach (var result in searchResults)
            {
                Console.WriteLine(result);
            }
            Console.ReadLine();
        }

        static List<string> ReadLinesFromFile(string filePath)
        {
            List<string> lines = new List<string>();
            try
            {
                lines = File.ReadAllLines(filePath).ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error reading file: " + ex.Message);
            }
            return lines;
        }

        static List<string> SemanticSearch(List<string> lines, string query)
        {
            var mlContext = new MLContext();

            // Metin verilerini yükleyin
            var data = lines.Select(line => new InputData { Text = line }).ToList();
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // TF-IDF pipeline'ı oluşturun
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text");

            // Modeli eğitin
            var model = pipeline.Fit(dataView);

            // Sorguyu vektörleştirin
            var queryData = new List<InputData> { new InputData { Text = query } };
            var queryDataView = mlContext.Data.LoadFromEnumerable(queryData);
            var queryTransformed = model.Transform(queryDataView);

            // Tüm satırları vektörleştirin
            var transformedData = model.Transform(dataView);

            // Vektörleri al
            var queryFeatures = mlContext.Data.CreateEnumerable<TransformedData>(queryTransformed, reuseRowObject: false).First().Features;
            var allFeatures = mlContext.Data.CreateEnumerable<TransformedData>(transformedData, reuseRowObject: false).ToList();

            // Benzerlik hesaplama (cosine similarity)
            var results = new List<(string Text, float Score)>();
            foreach (var feature in allFeatures)
            {
                var score = CosineSimilarity(queryFeatures, feature.Features);
                results.Add((feature.Text, score));
            }

            // Skora göre sırala ve en benzer sonuçları döndür
            return results.OrderByDescending(x => x.Score).Select(x => x.Text).Take(1).ToList();
        }

        static float CosineSimilarity(float[] vector1, float[] vector2)
        {
            float dotProduct = 0;
            float magnitude1 = 0;
            float magnitude2 = 0;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            magnitude1 = (float)Math.Sqrt(magnitude1);
            magnitude2 = (float)Math.Sqrt(magnitude2);

            if (magnitude1 * magnitude2 == 0) return 0;

            return dotProduct / (magnitude1 * magnitude2);
        }

        public class InputData
        {
            public string Text { get; set; }
        }

        public class TransformedData : InputData
        {
            [VectorType]
            public float[] Features { get; set; }
        }
    }
}