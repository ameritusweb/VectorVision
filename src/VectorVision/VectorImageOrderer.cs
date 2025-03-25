using ParallelReverseAutoDiff.PRAD;
using VectorVision.Tools;

namespace VectorVision
{
    public class VectorImageOrderer
    {
        private readonly VectorTools _vectorTools;
        private readonly ImageVectorConverter _imageConverter;
        private readonly int _vectorSize;
        private SharedWeightCoordinator _encodingCoordinator;
        private SharedWeight _encodingMatrixOp;  // For initial image encoding
        private SharedWeight _encodingWeightsOp;
        private PradOp _orderingMatrixOp;  // For ordering the encoded images
        private PradOp _orderingWeightsOp;

        public VectorImageOrderer(int vectorSize = 400, double learningRate = 0.0002)
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
            _encodingMatrixOp = new SharedWeight(Tensor.XavierUniform(new int[] { halfVectorSize, _vectorSize }));
            _encodingWeightsOp = new SharedWeight(Tensor.XavierUniform(new int[] { halfVectorSize, halfVectorSize }));
            _encodingCoordinator = new SharedWeightCoordinator();
            _encodingCoordinator.RegisterSharedWeight(_encodingMatrixOp);
            _encodingCoordinator.RegisterSharedWeight(_encodingWeightsOp);

            // Weights for ordering the encoded images
            _orderingMatrixOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, _vectorSize }));
            _orderingWeightsOp = new PradOp(Tensor.XavierUniform(new int[] { halfVectorSize, halfVectorSize }));
        }

        public void ResetGradients()
        {
            _encodingCoordinator.Reset();
            _orderingMatrixOp.ResetGradient();
            _orderingWeightsOp.ResetGradient();
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
                var pradOps = new List<PradOp>();
                for (int i = 0; i < imagePaths.Length; i++)
                {
                    // Convert image to initial vectors
                    var imageVectors = _imageConverter.ConvertImageToVectors(imagePaths[i]);
                    var imageVectorsOp = new PradOp(imageVectors);

                    var matrixOp = _encodingMatrixOp.UseOpAtIndex(i);
                    var weightsOp = _encodingWeightsOp.UseOpAtIndex(i);

                    // Apply vector-based matrix multiplication
                    var multiplied = _vectorTools.VectorBasedMatrixMultiplication(
                        imageVectorsOp,
                        matrixOp,
                        weightsOp
                    );

                    // Sum rows and transpose
                    var summed = _vectorTools.VectorSum2D(multiplied.PradOp);
                    var encoded = _vectorTools.VectorBasedTranspose(summed.PradOp);

                    _encodingCoordinator.RegisterResult(encoded);
                    pradOps.Add(encoded.PradOp);
                    imageEncodings.Add(encoded.Result);
                    encodedImages[i] = encoded.Result;
                }

                // Step 2: Concatenate all encoded images along axis 0
                var concatenated = new Tensor(
                    new int[] { imagePaths.Length, _vectorSize },
                    imageEncodings.SelectMany(t => t.Data).ToArray()
                );
                var concatenatedOp = new PradOp(concatenated);

                // Step 3: Apply ordering transformation
                var orderingResult = _vectorTools.VectorBasedMatrixMultiplication(
                    concatenatedOp,
                    _orderingMatrixOp,
                    _orderingWeightsOp
                );

                // Step 4: Order the results according to target order and create full sequence with reference vectors
                var orderedResults = new List<double>();
                for (int i = 0; i < targetOrder.Length; i++)
                {
                    int idx = targetOrder[i];
                    orderedResults.AddRange(orderingResult.Result.Data.Skip(idx * _vectorSize).Take(_vectorSize));
                }

                // Step 4: Create full sequence with reference vectors
                var fullSequence = startVector
                    .Concat(orderedResults)
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
                var orderedGradient = new Tensor(
                    new int[] { imagePaths.Length, _vectorSize },
                    sequenceOp.SeedGradient.Data.Skip(_vectorSize).Take(midValuesCount).ToArray()
                );

                // Unorder the gradient to match original sequence
                var unorderedGradient = new double[midValuesCount];
                for (int i = 0; i < targetOrder.Length; i++)
                {
                    int originalIdx = targetOrder[i];
                    Array.Copy(
                        orderedGradient.Data, i * _vectorSize,
                        unorderedGradient, originalIdx * _vectorSize,
                        _vectorSize
                    );
                }

                var orderingGradient = new Tensor(
                    new int[] { imagePaths.Length, _vectorSize },
                    unorderedGradient
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

                var concatenatedSeedGradient = concatenatedOp.SeedGradient;

                List<Tensor> concatenatedUpstreamGradientList = new List<Tensor>();

                // Split the concatenated gradient into individual tensors
                for (int i = 0; i < imagePaths.Length; i++)
                {
                    // Extract the gradient for this image
                    var imageGradient = new double[_vectorSize];
                    Array.Copy(
                        concatenatedSeedGradient.Data,
                        i * _vectorSize,
                        imageGradient,
                        0,
                        _vectorSize
                    );

                    // Create a new tensor with the extracted gradient
                    var gradientTensor = new Tensor(
                        new int[] { 1, _vectorSize },
                        imageGradient
                    );

                    concatenatedUpstreamGradientList.Add(gradientTensor);
                }

                _encodingCoordinator.BackpropagateAll(concatenatedUpstreamGradientList);

                // Update encoding weights
                var updatedEncoding = _vectorTools.GradientDescent(
                    _encodingMatrixOp.SharedOp,
                    _encodingWeightsOp.SharedOp
                );
                _encodingMatrixOp.Reset(updatedEncoding[0].CurrentTensor);
                _encodingWeightsOp.Reset(updatedEncoding[1].CurrentTensor);

                if (progressCallback != null && iter % 10 == 0)
                {
                    progressCallback(iter, finalLoss);
                }
            }

            return (finalLoss, encodedImages);
        }
    }
}