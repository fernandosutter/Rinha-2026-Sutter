using System.Diagnostics;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

const int Dims = 14;

string inputPath = args.Length > 0 ? args[0] : "/data/references.json.gz";
string outputPath = args.Length > 1 ? args[1] : "/data/index.bin";

Console.WriteLine($"Loading references from {inputPath}...");
var sw = Stopwatch.StartNew();

Stream stream = inputPath.EndsWith(".gz")
    ? new GZipStream(File.OpenRead(inputPath), CompressionMode.Decompress)
    : File.OpenRead(inputPath);

ReferenceEntry[]? refs;
using (stream)
{
    refs = JsonSerializer.Deserialize(stream, PreprocessorJsonContext.Default.ReferenceEntryArray);
}

if (refs is null || refs.Length == 0)
{
    Console.Error.WriteLine("Failed to load references.");
    return 1;
}

int totalVectors = refs.Length;
Console.WriteLine($"Loaded {totalVectors} vectors in {sw.Elapsed.TotalSeconds:F1}s");

// Extract float vectors and labels
var vectors = new float[totalVectors * Dims];
var labels = new byte[totalVectors];

for (int i = 0; i < totalVectors; i++)
{
    var vec = refs[i].Vector;
    for (int d = 0; d < Dims; d++)
        vectors[i * Dims + d] = vec[d];
    labels[i] = refs[i].Label == "fraud" ? (byte)1 : (byte)0;
}

refs = null; // Release memory
GC.Collect();

// K-Means clustering
int nClusters = Math.Min(3000, totalVectors / 100);
if (nClusters < 1) nClusters = 1;

Console.WriteLine($"Running Mini-Batch K-Means++ with {nClusters} clusters...");
sw.Restart();

var centroids = new float[nClusters * Dims];
var rng = new Random(42);

// KMeans++ initialization: pick first centroid randomly, then pick subsequent
// centroids with probability proportional to squared distance from nearest existing centroid
{
    int firstIdx = rng.Next(totalVectors);
    Array.Copy(vectors, firstIdx * Dims, centroids, 0, Dims);

    // Pre-compute distances for KMeans++ (use a sample for speed with 3M vectors)
    int sampleSize = Math.Min(100_000, totalVectors);
    var sampleIndices = new int[sampleSize];
    for (int i = 0; i < sampleSize; i++)
        sampleIndices[i] = rng.Next(totalVectors);

    var minDists = new float[sampleSize];
    Array.Fill(minDists, float.MaxValue);

    for (int c = 1; c < nClusters; c++)
    {
        // Update min distances to the last added centroid
        var lastCentroid = centroids.AsSpan((c - 1) * Dims, Dims);
        double sumDists = 0;
        for (int s = 0; s < sampleSize; s++)
        {
            int vecIdx = sampleIndices[s];
            float dist = 0;
            int offset = vecIdx * Dims;
            for (int d = 0; d < Dims; d++)
            {
                float diff = vectors[offset + d] - lastCentroid[d];
                dist += diff * diff;
            }
            if (dist < minDists[s])
                minDists[s] = dist;
            sumDists += minDists[s];
        }

        // Pick next centroid with probability proportional to distance²
        double threshold = rng.NextDouble() * sumDists;
        double cumulative = 0;
        int chosen = sampleIndices[sampleSize - 1];
        for (int s = 0; s < sampleSize; s++)
        {
            cumulative += minDists[s];
            if (cumulative >= threshold)
            {
                chosen = sampleIndices[s];
                break;
            }
        }
        Array.Copy(vectors, chosen * Dims, centroids, c * Dims, Dims);

        if (c % 500 == 0)
            Console.WriteLine($"  KMeans++ init: {c}/{nClusters} centroids");
    }
}
Console.WriteLine($"KMeans++ initialization done in {sw.Elapsed.TotalSeconds:F1}s");

// Mini-batch K-Means iterations
int batchSize = Math.Min(20000, totalVectors);
int iterations = 120;

sw.Restart();
for (int iter = 0; iter < iterations; iter++)
{
    var clusterCounts = new int[nClusters];
    var clusterSums = new double[nClusters * Dims];

    for (int b = 0; b < batchSize; b++)
    {
        int vecIdx = rng.Next(totalVectors);
        var vecSpan = vectors.AsSpan(vecIdx * Dims, Dims);

        int bestCluster = FindNearest(vecSpan, centroids, nClusters);
        clusterCounts[bestCluster]++;
        for (int d = 0; d < Dims; d++)
            clusterSums[bestCluster * Dims + d] += vecSpan[d];
    }

    float lr = 1f / (iter + 2); // +2 for more stable convergence
    for (int c = 0; c < nClusters; c++)
    {
        if (clusterCounts[c] == 0) continue;
        for (int d = 0; d < Dims; d++)
        {
            float mean = (float)(clusterSums[c * Dims + d] / clusterCounts[c]);
            centroids[c * Dims + d] = centroids[c * Dims + d] * (1 - lr) + mean * lr;
        }
    }

    if ((iter + 1) % 20 == 0)
        Console.WriteLine($"  Iteration {iter + 1}/{iterations}");
}

Console.WriteLine($"K-Means done in {sw.Elapsed.TotalSeconds:F1}s");

// Full assignment
Console.WriteLine("Assigning all vectors to clusters...");
sw.Restart();

var assignments = new int[totalVectors];
Parallel.For(0, totalVectors, i =>
{
    assignments[i] = FindNearest(vectors.AsSpan(i * Dims, Dims), centroids, nClusters);
});

Console.WriteLine($"Assignment done in {sw.Elapsed.TotalSeconds:F1}s");

// Sort by cluster
Console.WriteLine("Building index...");
sw.Restart();

var sortedIndices = Enumerable.Range(0, totalVectors)
    .OrderBy(i => assignments[i])
    .ToArray();

var clusterOffsets = new int[nClusters + 1];
var halfVectors = new Half[totalVectors * Dims];
var sortedLabels = new byte[totalVectors];

for (int i = 0; i < totalVectors; i++)
{
    int origIdx = sortedIndices[i];
    for (int d = 0; d < Dims; d++)
        halfVectors[i * Dims + d] = (Half)vectors[origIdx * Dims + d];
    sortedLabels[i] = labels[origIdx];
}

// Build offsets
int currentCluster = 0;
clusterOffsets[0] = 0;
for (int i = 0; i < totalVectors; i++)
{
    int cluster = assignments[sortedIndices[i]];
    while (currentCluster < cluster)
    {
        currentCluster++;
        clusterOffsets[currentCluster] = i;
    }
}
while (currentCluster < nClusters)
{
    currentCluster++;
    clusterOffsets[currentCluster] = totalVectors;
}

Console.WriteLine($"Index built in {sw.Elapsed.TotalSeconds:F1}s");

// Write binary
Console.WriteLine($"Writing to {outputPath}...");
sw.Restart();

using (var fs = File.Create(outputPath))
using (var bw = new BinaryWriter(fs))
{
    bw.Write(nClusters);
    bw.Write(totalVectors);
    bw.Write(Dims);

    for (int i = 0; i < centroids.Length; i++)
        bw.Write(centroids[i]);

    for (int i = 0; i < clusterOffsets.Length; i++)
        bw.Write(clusterOffsets[i]);

    var vecBytes = MemoryMarshal.AsBytes(halfVectors.AsSpan()).ToArray();
    bw.Write(vecBytes);

    bw.Write(sortedLabels);
}

var fileSize = new FileInfo(outputPath).Length;
Console.WriteLine($"Wrote {fileSize / 1024 / 1024}MB to {outputPath} in {sw.Elapsed.TotalSeconds:F1}s");
Console.WriteLine("Preprocessor done!");

return 0;

static int FindNearest(ReadOnlySpan<float> query, float[] centroids, int nClusters)
{
    int best = 0;
    float bestDist = float.MaxValue;
    for (int c = 0; c < nClusters; c++)
    {
        float dist = 0;
        int offset = c * 14;
        for (int d = 0; d < 14; d++)
        {
            float diff = query[d] - centroids[offset + d];
            dist += diff * diff;
        }
        if (dist < bestDist)
        {
            bestDist = dist;
            best = c;
        }
    }
    return best;
}

public sealed class ReferenceEntry
{
    [JsonPropertyName("vector")]
    public float[] Vector { get; set; } = [];

    [JsonPropertyName("label")]
    public string Label { get; set; } = "";
}

[JsonSerializable(typeof(ReferenceEntry[]))]
internal partial class PreprocessorJsonContext : JsonSerializerContext
{
}
