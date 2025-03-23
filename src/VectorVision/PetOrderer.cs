using ParallelReverseAutoDiff.PRAD;

namespace VectorVision
{
    public class PetOrderer
    {
        private readonly VectorImageOrderer _orderer;
        private readonly List<string> _allImagePaths;
        private readonly int _batchSize;
        private readonly List<double> _lossHistory = new List<double>();
        private readonly ImageVectorConverter _imageConverter = new ImageVectorConverter();
        private readonly Random _random = new Random(42); // Fixed seed for reproducibility

        public PetOrderer(string[] imagePaths, int batchSize, int vectorSize = 48)
        {
            _orderer = new VectorImageOrderer(vectorSize: vectorSize, learningRate: 0.0002);
            _allImagePaths = new List<string>(imagePaths);
            _batchSize = batchSize;
        }

        public async Task<double> Train(int epochs = 10, bool logProgress = true)
        {
            Console.WriteLine($"Starting training with {_allImagePaths.Count} total images");
            Console.WriteLine($"Batch size: {_batchSize}");
            Console.WriteLine($"Epochs: {epochs}\n");

            double finalLoss = 0;
            int totalIterations = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Shuffle all image paths at the start of each epoch
                var shuffledPaths = new List<string>(_allImagePaths);
                Shuffle(shuffledPaths);

                Console.WriteLine($"Epoch {epoch + 1}/{epochs}");

                while (shuffledPaths.Count >= _batchSize)
                {
                    // Take next batch of images
                    var batchPaths = shuffledPaths.Take(_batchSize).ToArray();
                    shuffledPaths.RemoveRange(0, _batchSize);

                    // Get target ordering for this batch
                    int[] targetOrder = CreateLightnessSortedOrder(batchPaths);

                    // Train on this batch
                    var (loss, _) = _orderer.TrainOrdering(
                        batchPaths,
                        targetOrder,
                        iterations: 1, // One iteration per batch
                        (iter, currentLoss) =>
                        {
                            _lossHistory.Add(currentLoss);
                            if (logProgress)
                            {
                                Console.WriteLine($"Batch {totalIterations + 1}: Loss = {currentLoss:F6}");
                                Console.WriteLine($"Images remaining in epoch: {shuffledPaths.Count}");
                            }
                        }
                    );

                    finalLoss = loss;
                    totalIterations++;
                }

                // Handle remaining images if any (less than batch size)
                if (shuffledPaths.Count > 0)
                {
                    Console.WriteLine($"Processing remaining {shuffledPaths.Count} images in epoch");
                    var batchPaths = shuffledPaths.ToArray();
                    var targetOrder = CreateLightnessSortedOrder(batchPaths);
                    var (loss, _) = _orderer.TrainOrdering(batchPaths, targetOrder, iterations: 1);
                    finalLoss = loss;
                    totalIterations++;
                }

                Console.WriteLine($"Epoch {epoch + 1} completed. Current loss: {finalLoss:F6}\n");
            }

            // Save training history
            await File.WriteAllLinesAsync(
                "loss_history.csv",
                new[] { "Iteration,Loss" }.Concat(
                    _lossHistory.Select((loss, i) => $"{i},{loss}")
                )
            );

            Console.WriteLine("Training completed!");
            Console.WriteLine($"Initial loss: {_lossHistory[0]:F6}");
            Console.WriteLine($"Final loss: {finalLoss:F6}");
            Console.WriteLine($"Loss reduction: {((_lossHistory[0] - finalLoss) / _lossHistory[0] * 100):F2}%");
            Console.WriteLine($"Total iterations: {totalIterations}");

            return finalLoss;
        }

        private void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = _random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        private double CalculateAverageLightness(string imagePath)
        {
            var vectors = _imageConverter.ConvertImageToVectors(imagePath);
            int halfCols = vectors.Shape[1] / 2;
            double sum = 0;
            int count = 0;

            for (int i = 0; i < vectors.Shape[0]; i++)
            {
                for (int j = 0; j < halfCols; j++)
                {
                    sum += vectors[i, j];
                    count++;
                }
            }

            return sum / count;
        }

        private int[] CreateLightnessSortedOrder(string[] batchPaths)
        {
            // Create tuples of (index, isCat, lightness)
            var imageInfo = batchPaths.Select((path, index) => (
                Index: index,
                IsCat: path.ToLower().Contains("cat"),
                Lightness: CalculateAverageLightness(path)
            )).ToList();

            // Order cats first (by lightness), then dogs (by lightness)
            return imageInfo
                .OrderByDescending(x => x.IsCat)  // Cats first
                .ThenBy(x => x.Lightness)         // Sort by lightness within each group
                .Select(x => x.Index)
                .ToArray();
        }

        public double[] GetLossHistory() => _lossHistory.ToArray();
    }
}