namespace VectorVision
{
    using Emgu.CV;
    using Emgu.CV.CvEnum;
    using Emgu.CV.Structure;
    using ParallelReverseAutoDiff.PRAD;

    public class ImageVectorConverter
    {
        /// <summary>
        /// Converts an image to a vector representation where magnitude represents lightness and angle represents hue
        /// </summary>
        /// <param name="imagePath">Path to the PNG image</param>
        /// <returns>Tensor with magnitudes (lightness) in left half and angles (hue) in right half</returns>
        public Tensor ConvertImageToVectors(string imagePath)
        {
            // Load the image
            using var image = CvInvoke.Imread(imagePath, ImreadModes.Color);

            // Convert to HSV color space
            using var hsvImage = new Mat();
            CvInvoke.CvtColor(image, hsvImage, ColorConversion.Bgr2Hsv);

            // Get image data
            var hsvData = hsvImage.ToImage<Hsv, byte>();

            int rows = hsvImage.Height;
            int cols = hsvImage.Width;

            // Create arrays for magnitudes (lightness) and angles (hue)
            double[,] magnitudes = new double[rows, cols];
            double[,] angles = new double[rows, cols];

            // Process each pixel
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
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
            int tensorCols = cols * 2;
            double[,] combinedData = new double[rows, tensorCols];

            // Copy magnitudes to left half
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    combinedData[i, j] = magnitudes[i, j];
                    combinedData[i, j + cols] = angles[i, j];
                }
            }

            return new Tensor(combinedData);
        }

        public void ConvertVectorsToImage(Tensor vectorTensor, string outputPath)
        {
            int rows = vectorTensor.Shape[0];
            int totalCols = vectorTensor.Shape[1];
            int cols = totalCols / 2;

            // Create HSV image
            var hsvImage = new Image<Hsv, byte>(cols, rows);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    // Get magnitude (lightness) and angle (hue)
                    double magnitude = vectorTensor[i, j];
                    double angle = vectorTensor[i, j + cols];

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