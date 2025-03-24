namespace VectorVision
{
    using Emgu.CV;
    using Emgu.CV.CvEnum;
    using Emgu.CV.Structure;
    using ParallelReverseAutoDiff.PRAD;

    public class ImageVectorConverter
    {
        private const int TargetSize = 200;

        /// <summary>
        /// Converts an image to a vector representation where magnitude represents lightness and angle represents hue
        /// </summary>
        /// <param name="imagePath">Path to the PNG image</param>
        /// <returns>Tensor with magnitudes (lightness) in left half and angles (hue) in right half</returns>
        public Tensor ConvertImageToVectors(string imagePath)
        {
            // Load the image
            using var image = CvInvoke.Imread(imagePath, ImreadModes.Color);

            // Resize image to 200x200
            using var resizedImage = new Mat();
            CvInvoke.Resize(image, resizedImage, new System.Drawing.Size(TargetSize, TargetSize), 0, 0, Inter.Lanczos4);

            // Convert to HSV color space
            using var hsvImage = new Mat();
            CvInvoke.CvtColor(resizedImage, hsvImage, ColorConversion.Bgr2Hsv);

            // Get image data
            var hsvData = hsvImage.ToImage<Hsv, byte>();

            // Create arrays for magnitudes (lightness) and angles (hue)
            double[,] magnitudes = new double[TargetSize, TargetSize];
            double[,] angles = new double[TargetSize, TargetSize];

            // Process each pixel
            for (int i = 0; i < TargetSize; i++)
            {
                for (int j = 0; j < TargetSize; j++)
                {
                    // Get HSV values
                    var pixel = hsvData[i, j];

                    // Convert hue to radians (OpenCV uses 0-180 for hue)
                    double hueRadians = (pixel.Hue * 2.0 * Math.PI) / 180.0;

                    // Normalize value to 0-1 range for magnitude
                    double normalizedValue = pixel.Value / 255.0;

                    // Store in arrays
                    magnitudes[i, j] = normalizedValue;
                    angles[i, j] = hueRadians;
                }
            }

            // Create combined tensor
            int tensorCols = TargetSize * 2;
            double[,] combinedData = new double[TargetSize, tensorCols];

            // Copy magnitudes to left half and angles to right half
            for (int i = 0; i < TargetSize; i++)
            {
                for (int j = 0; j < TargetSize; j++)
                {
                    combinedData[i, j] = magnitudes[i, j];
                    combinedData[i, j + TargetSize] = angles[i, j];
                }
            }

            return new Tensor(combinedData);
        }

        public void ConvertVectorsToImage(Tensor vectorTensor, string outputPath)
        {
            if (vectorTensor.Shape[0] != TargetSize || vectorTensor.Shape[1] != TargetSize * 2)
            {
                throw new ArgumentException($"Input tensor must have shape [{TargetSize}, {TargetSize * 2}]");
            }

            // Create HSV image
            var hsvImage = new Image<Hsv, byte>(TargetSize, TargetSize);

            for (int i = 0; i < TargetSize; i++)
            {
                for (int j = 0; j < TargetSize; j++)
                {
                    // Get magnitude (lightness) and angle (hue)
                    double magnitude = vectorTensor[i, j];
                    double angle = vectorTensor[i, j + TargetSize];

                    // Convert angle back to OpenCV hue range (0-180)
                    byte hue = (byte)((angle * 180.0) / (2.0 * Math.PI));

                    // Convert magnitude back to value range (0-255)
                    byte value = (byte)(magnitude * 255.0);

                    // Create HSV pixel
                    hsvImage[i, j] = new Hsv(hue, 255, value);
                }
            }

            // Convert back to BGR and save
            using var bgrImage = hsvImage.Convert<Bgr, byte>();
            bgrImage.Save(outputPath);
        }
    }
}