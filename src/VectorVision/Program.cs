namespace VectorVision
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            // Get all pet images
            string[] imagePaths = Directory.GetFiles("E:\\PetImages\\Animals", "*.jpg", SearchOption.AllDirectories);

            // Create POC with batch size of 5
            var poc = new PetOrderer(
                imagePaths: imagePaths,
                batchSize: 5,
                vectorSize: 48
            );

            // Train for 10 epochs
            await poc.Train(epochs: 10);
        }
    }
}
