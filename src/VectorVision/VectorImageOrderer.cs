using ParallelReverseAutoDiff.PRAD;
using VectorVision.Tools;

namespace VectorVision
{
    public class VectorImageOrderer
    {
        private readonly VectorTools _vectorTools;
        private readonly ImageVectorConverter _imageConverter;
        private readonly int _vectorSize;
        private PradOp _encodingMatrixOp;  // For initial image encoding
        private PradOp _encodingWeightsOp;
        private PradOp _orderingMatrixOp;  // For ordering the encoded images
        private PradOp _orderingWeightsOp;

        public VectorImageOrderer(int vectorSize = 48, double learningRate = 0.0002)
        {
            _vectorTools = new VectorTools { LearningRate = learningRate };
            _imageConverter = new ImageVectorConverter();
            _vectorSize = vectorSize;
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            int halfVectorSize = _vectorSize / 2;

            // Weights for initial image encoding
            _encodingMatrixOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, _vectorSize }));
            _encodingWeightsOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, halfVectorSize }));

            // Weights for ordering the encoded images
            _orderingMatrixOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, _vectorSize }));
            _orderingWeightsOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, halfVectorSize }));
        }

        public (double loss, Tensor[] encodedImages) TrainOrdering(
            string[] imagePaths,
            int[] targetOrder, // Add target ordering parameter
            int iterations = 500,
            Action<int, double>? progressCallback = null)
        {
            // Create reference vectors for start and end
            double[] startVector = Enumerable.Repeat(0.5, _vectorSize / 2)
                                          .Concat(Enumerable.Repeat(0.5, _vectorSize / 2))
                                          .ToArray();
            double[] endVector = Enumerable.Repeat(0.5, _vectorSize / 2)
                                        .Concat(Enumerable.Repeat(-0.5, _vectorSize / 2))
                                        .ToArray();

            // Keep track of encoded images
            Tensor[] encodedImages = new Tensor[imagePaths.Length];
            double finalLoss = 0;

            for (int iter = 0; iter < iterations; iter++)
            {
                // Step 1: Encode each image using vector operations
                var imageEncodings = new List<Tensor>();
                for (int i = 0; i < imagePaths.Length; i++)
                {
                    // Convert image to initial vectors
                    var imageVectors = _imageConverter.ConvertImageToVectors(imagePaths[i]);
                    var imageVectorsOp = new PradOp(imageVectors);

                    // Apply vector-based matrix multiplication
                    var multiplied = _vectorTools.VectorBasedMatrixMultiplication(
                        imageVectorsOp,
                        _encodingMatrixOp,
                        _encodingWeightsOp
                    );

                    // Sum rows and transpose
                    var summed = _vectorTools.VectorSum2D(multiplied.PradOp);
                    var encoded = _vectorTools.VectorBasedTranspose(summed.PradOp);

                    imageEncodings.Add(encoded.Result);
                    encodedImages[i] = encoded.Result;
                }

                // Step 2: Concatenate all encoded images along axis 0 in target order
                var orderedEncodings = new List<Tensor>();
                foreach (int idx in targetOrder)
                {
                    orderedEncodings.Add(imageEncodings[idx]);
                }

                var concatenated = new Tensor(
                    new int[] { imagePaths.Length, _vectorSize },
                    orderedEncodings.SelectMany(t => t.Data).ToArray()
                );
                var concatenatedOp = new PradOp(concatenated);

                // Step 3: Apply ordering transformation
                var orderingResult = _vectorTools.VectorBasedMatrixMultiplication(
                    concatenatedOp,
                    _orderingMatrixOp,
                    _orderingWeightsOp
                );

                // Step 4: Create full sequence with reference vectors
                var fullSequence = startVector
                    .Concat(orderingResult.Result.Data)
                    .Concat(endVector)
                    .ToArray();
                var sequenceOp = new PradOp(new Tensor(
                    new int[] { imagePaths.Length + 2, _vectorSize },
                    fullSequence
                ));

                // Step 5: Compute ordering loss
                var lossResult = _vectorTools.ComputeOrderingLoss2(sequenceOp);
                finalLoss = lossResult.Result[0, 0];

                // Step 6: Backpropagation
                // Start with loss gradient
                var upstream = new Tensor(new int[] { 1, 1 }, 1d);
                lossResult.Back(upstream);

                // Extract gradients for the middle section (excluding reference vectors)
                int midValuesCount = imagePaths.Length * _vectorSize;
                var orderingGradient = new Tensor(
                    new int[] { imagePaths.Length, _vectorSize },
                    sequenceOp.SeedGradient.Data.Skip(_vectorSize).Take(midValuesCount).ToArray()
                );

                // Backpropagate through ordering transformation
                orderingResult.PradOp.Back(orderingGradient);

                // Update weights using gradient descent
                var updatedOrdering = _vectorTools.GradientDescent(
                    _orderingMatrixOp,
                    _orderingWeightsOp
                );
                _orderingMatrixOp = updatedOrdering[0];
                _orderingWeightsOp = updatedOrdering[1];

                // Update encoding weights
                var updatedEncoding = _vectorTools.GradientDescent(
                    _encodingMatrixOp,
                    _encodingWeightsOp
                );
                _encodingMatrixOp = updatedEncoding[0];
                _encodingWeightsOp = updatedEncoding[1];

                if (progressCallback != null && iter % 10 == 0)
                {
                    progressCallback(iter, finalLoss);
                }
            }

            return (finalLoss, encodedImages);
        }

        public Tensor EncodeImage(string imagePath)
        {
            var imageVectors = _imageConverter.ConvertImageToVectors(imagePath);
            var imageVectorsOp = new PradOp(imageVectors);

            var multiplied = _vectorTools.VectorBasedMatrixMultiplication(
                imageVectorsOp,
                _encodingMatrixOp,
                _encodingWeightsOp
            );

            var summed = _vectorTools.VectorSum2D(multiplied.PradOp);
            var encoded = _vectorTools.VectorBasedTranspose(summed.PradOp);

            return encoded.Result;
        }

        public double[] GetEncodingWeights() => _encodingWeightsOp.CurrentTensor.Data;
        public double[] GetOrderingWeights() => _orderingWeightsOp.CurrentTensor.Data;
    }
}