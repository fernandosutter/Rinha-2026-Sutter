using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Rinha_2026_WebAPI.Services;

public sealed class IvfIndex
{
    private const int Dims = 14;
    private const int K = 5; // k nearest neighbors

    // IVF structures
    private float[] _centroids = []; // nClusters * Dims
    private int _nClusters;
    private int[] _clusterOffsets = []; // nClusters + 1
    private Half[] _vectors = []; // totalVectors * Dims (sorted by cluster)
    private byte[] _labels = []; // totalVectors (0=legit, 1=fraud)
    private int _totalVectors;
    private int _nprobe = 8;

    public bool IsLoaded => _totalVectors > 0;

    public void LoadFromBinary(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        _nClusters = br.ReadInt32();
        _totalVectors = br.ReadInt32();
        int dims = br.ReadInt32();

        // Centroids: nClusters * dims * float32
        _centroids = new float[_nClusters * dims];
        for (int i = 0; i < _centroids.Length; i++)
            _centroids[i] = br.ReadSingle();

        // Offsets: (nClusters + 1) * int32
        _clusterOffsets = new int[_nClusters + 1];
        for (int i = 0; i < _clusterOffsets.Length; i++)
            _clusterOffsets[i] = br.ReadInt32();

        // Vectors: totalVectors * dims * Half (2 bytes)
        _vectors = new Half[_totalVectors * dims];
        br.BaseStream.ReadExactly(MemoryMarshal.AsBytes(_vectors.AsSpan()));

        // Labels: totalVectors * byte
        _labels = br.ReadBytes(_totalVectors);
    }

    public void LoadFromJson(string path)
    {
        Stream stream;
        if (path.EndsWith(".gz"))
        {
            stream = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
        }
        else
        {
            stream = File.OpenRead(path);
        }

        using (stream)
        {
            var refs = JsonSerializer.Deserialize(stream, RefJsonContext.Default.ReferenceEntryArray);
            if (refs is null) return;

            _totalVectors = refs.Length;

            // Store vectors as float temporarily for clustering
            var floatVectors = new float[_totalVectors * Dims];
            _labels = new byte[_totalVectors];

            for (int i = 0; i < _totalVectors; i++)
            {
                var vec = refs[i].Vector;
                for (int d = 0; d < Dims; d++)
                    floatVectors[i * Dims + d] = vec[d];
                _labels[i] = refs[i].Label == "fraud" ? (byte)1 : (byte)0;
            }

            // Build IVF index
            BuildIvf(floatVectors);
        }
    }

    private void BuildIvf(float[] allVectors)
    {
        _nClusters = Math.Min(3000, _totalVectors / 100);
        if (_nClusters < 1) _nClusters = 1;

        // KMeans++ initialization
        _centroids = new float[_nClusters * Dims];
        var rng = new Random(42);

        // Pick first centroid randomly
        int firstIdx = rng.Next(_totalVectors);
        Array.Copy(allVectors, firstIdx * Dims, _centroids, 0, Dims);

        // Subsequent centroids: probability proportional to distance²
        int sampleSize = Math.Min(50_000, _totalVectors);
        var sampleIndices = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++)
            sampleIndices[i] = rng.Next(_totalVectors);

        var minDists = new float[sampleSize];
        Array.Fill(minDists, float.MaxValue);

        for (int c = 1; c < _nClusters; c++)
        {
            var lastCentroid = _centroids.AsSpan((c - 1) * Dims, Dims);
            double sumDists = 0;
            for (int s = 0; s < sampleSize; s++)
            {
                int vecIdx = sampleIndices[s];
                float dist = 0;
                for (int d = 0; d < Dims; d++)
                {
                    float diff = allVectors[vecIdx * Dims + d] - lastCentroid[d];
                    dist += diff * diff;
                }
                if (dist < minDists[s])
                    minDists[s] = dist;
                sumDists += minDists[s];
            }

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
            Array.Copy(allVectors, chosen * Dims, _centroids, c * Dims, Dims);
        }

        // Assignments
        var assignments = new int[_totalVectors];
        int batchSize = Math.Min(20000, _totalVectors);
        int iterations = 100;

        for (int iter = 0; iter < iterations; iter++)
        {
            // Mini-batch: random subset
            var batchIndices = new int[batchSize];
            for (int b = 0; b < batchSize; b++)
                batchIndices[b] = rng.Next(_totalVectors);

            // Assign batch to nearest centroid
            var clusterCounts = new int[_nClusters];
            var clusterSums = new double[_nClusters * Dims];

            for (int b = 0; b < batchSize; b++)
            {
                int vecIdx = batchIndices[b];
                var vecSpan = allVectors.AsSpan(vecIdx * Dims, Dims);
                int bestCluster = FindNearestCentroid(vecSpan);
                clusterCounts[bestCluster]++;
                for (int d = 0; d < Dims; d++)
                    clusterSums[bestCluster * Dims + d] += vecSpan[d];
            }

            // Update centroids (moving average)
            float lr = 1f / (iter + 2);
            for (int c = 0; c < _nClusters; c++)
            {
                if (clusterCounts[c] == 0) continue;
                for (int d = 0; d < Dims; d++)
                {
                    float mean = (float)(clusterSums[c * Dims + d] / clusterCounts[c]);
                    _centroids[c * Dims + d] = _centroids[c * Dims + d] * (1 - lr) + mean * lr;
                }
            }
        }

        // Full assignment pass
        for (int i = 0; i < _totalVectors; i++)
        {
            assignments[i] = FindNearestCentroid(allVectors.AsSpan(i * Dims, Dims));
        }

        // Sort by cluster and build offsets
        var sortedIndices = Enumerable.Range(0, _totalVectors)
            .OrderBy(i => assignments[i])
            .ToArray();

        _clusterOffsets = new int[_nClusters + 1];
        _vectors = new Half[_totalVectors * Dims];
        var sortedLabels = new byte[_totalVectors];

        for (int i = 0; i < _totalVectors; i++)
        {
            int origIdx = sortedIndices[i];
            for (int d = 0; d < Dims; d++)
                _vectors[i * Dims + d] = (Half)allVectors[origIdx * Dims + d];
            sortedLabels[i] = _labels[origIdx];
        }
        _labels = sortedLabels;

        // Build offsets
        int currentCluster = 0;
        _clusterOffsets[0] = 0;
        for (int i = 0; i < _totalVectors; i++)
        {
            int cluster = assignments[sortedIndices[i]];
            while (currentCluster < cluster)
            {
                currentCluster++;
                _clusterOffsets[currentCluster] = i;
            }
        }
        while (currentCluster < _nClusters)
        {
            currentCluster++;
            _clusterOffsets[currentCluster] = _totalVectors;
        }

        // Adjust nprobe based on cluster count. 8 keeps recall high while halving
        // candidate scan work vs 15 (matters a lot on throttled CPU).
        _nprobe = Math.Min(8, _nClusters);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int FindNearestCentroid(ReadOnlySpan<float> query)
    {
        int best = 0;
        float bestDist = float.MaxValue;
        for (int c = 0; c < _nClusters; c++)
        {
            float dist = SimdDistance.EuclideanDistanceSquared(query, _centroids.AsSpan(c * Dims, Dims));
            if (dist < bestDist)
            {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    public (float fraudScore, bool approved) Search(ReadOnlySpan<float> query)
    {
        // Find top nprobe nearest centroids
        Span<int> probeClusters = stackalloc int[_nprobe];
        Span<float> probeDists = stackalloc float[_nprobe];
        probeDists.Fill(float.MaxValue);

        for (int c = 0; c < _nClusters; c++)
        {
            float dist = SimdDistance.EuclideanDistanceSquared(query, _centroids.AsSpan(c * Dims, Dims));

            // Insert into sorted probe list (max-heap style: replace worst if better)
            if (dist < probeDists[_nprobe - 1])
            {
                probeDists[_nprobe - 1] = dist;
                probeClusters[_nprobe - 1] = c;

                // Bubble up
                for (int j = _nprobe - 1; j > 0; j--)
                {
                    if (probeDists[j] < probeDists[j - 1])
                    {
                        (probeDists[j], probeDists[j - 1]) = (probeDists[j - 1], probeDists[j]);
                        (probeClusters[j], probeClusters[j - 1]) = (probeClusters[j - 1], probeClusters[j]);
                    }
                    else break;
                }
            }
        }

        // Search within probe clusters, maintain top-K with early exit
        Span<float> topDists = stackalloc float[K];
        Span<byte> topLabels = stackalloc byte[K];
        topDists.Fill(float.MaxValue);

        int filledCount = 0; // track how many of K slots are filled

        for (int p = 0; p < _nprobe; p++)
        {
            int cluster = probeClusters[p];
            if (probeDists[p] == float.MaxValue) break;

            int start = _clusterOffsets[cluster];
            int end = _clusterOffsets[cluster + 1];

            for (int i = start; i < end; i++)
            {
                float dist = SimdDistance.EuclideanDistanceSquaredHalf(query, _vectors.AsSpan(i * Dims, Dims));

                if (dist < topDists[K - 1])
                {
                    topDists[K - 1] = dist;
                    topLabels[K - 1] = _labels[i];
                    if (filledCount < K) filledCount++;

                    // Insertion sort into top-K
                    for (int j = K - 1; j > 0; j--)
                    {
                        if (topDists[j] < topDists[j - 1])
                        {
                            (topDists[j], topDists[j - 1]) = (topDists[j - 1], topDists[j]);
                            (topLabels[j], topLabels[j - 1]) = (topLabels[j - 1], topLabels[j]);
                        }
                        else break;
                    }
                }
            }

            // Early exit: once K slots are filled and we already scanned >=2 probes,
            // if all current top-K agree, the decision is stable.
            if (filledCount >= K && p >= 1)
            {
                int fraudVotes = 0;
                for (int i = 0; i < K; i++)
                {
                    if (topLabels[i] == 1) fraudVotes++;
                }
                // Unanimous (all fraud or all legit) -> stop scanning further clusters
                if (fraudVotes == K || fraudVotes == 0)
                    break;
                // Also exit on dominant majority (4/5 same class) after 3 probes:
                // remaining clusters are unlikely to flip 4-vote majority and they
                // are farther centroids by definition.
                if (p >= 3 && (fraudVotes >= 4 || fraudVotes <= 1))
                    break;
            }
        }

        int fraudCount = 0;
        for (int i = 0; i < K; i++)
        {
            if (topLabels[i] == 1) fraudCount++;
        }

        float fraudScore = fraudCount / (float)K;
        bool approved = fraudScore < 0.6f;
        return (fraudScore, approved);
    }

    public void SaveToBinary(string path)
    {
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);

        bw.Write(_nClusters);
        bw.Write(_totalVectors);
        bw.Write(Dims);

        for (int i = 0; i < _centroids.Length; i++)
            bw.Write(_centroids[i]);

        for (int i = 0; i < _clusterOffsets.Length; i++)
            bw.Write(_clusterOffsets[i]);

        var vecBytes = new byte[_vectors.Length * 2];
        Buffer.BlockCopy(_vectors, 0, vecBytes, 0, vecBytes.Length);
        bw.Write(vecBytes);

        bw.Write(_labels);
    }
}

// JSON models for reference data
public sealed class ReferenceEntry
{
    [System.Text.Json.Serialization.JsonPropertyName("vector")]
    public float[] Vector { get; set; } = [];

    [System.Text.Json.Serialization.JsonPropertyName("label")]
    public string Label { get; set; } = "";
}

[System.Text.Json.Serialization.JsonSerializable(typeof(ReferenceEntry[]))]
internal partial class RefJsonContext : System.Text.Json.Serialization.JsonSerializerContext
{
}
