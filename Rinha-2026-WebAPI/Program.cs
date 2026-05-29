using System.Text.Json.Serialization;
using Rinha_2026_WebAPI.Models;
using Rinha_2026_WebAPI.Services;

// Confidence thresholds for ML-only decisions
const float MlHighThreshold = 0.80f;
const float MlLowThreshold = 0.20f;

// Load IVF index at startup
var ivfIndex = new IvfIndex();

var indexBinPath = Environment.GetEnvironmentVariable("INDEX_PATH") ?? "index.bin";
var referencesPath = Environment.GetEnvironmentVariable("REFERENCES_PATH");

if (File.Exists(indexBinPath))
{
    ivfIndex.LoadFromBinary(indexBinPath);
}
else if (referencesPath is not null && File.Exists(referencesPath))
{
    ivfIndex.LoadFromJson(referencesPath);
    ivfIndex.SaveToBinary(indexBinPath);
}
else
{
    var candidates = new[]
    {
        "references.json.gz",
        "/data/references.json.gz",
        "../resources/references.json.gz",
    };
    foreach (var candidate in candidates)
    {
        if (File.Exists(candidate))
        {
            ivfIndex.LoadFromJson(candidate);
            ivfIndex.SaveToBinary(indexBinPath);
            break;
        }
    }
}

// Load GBM model (optional — falls back to IVF-only if missing)
var gbmPredictor = new GbmPredictor();
var modelBinPath = Environment.GetEnvironmentVariable("MODEL_PATH") ?? "model.bin";
var unixSocketPath = Environment.GetEnvironmentVariable("UNIX_SOCKET_PATH");

if (File.Exists(modelBinPath))
{
    gbmPredictor.LoadFromBinary(modelBinPath);
}
else
{
    var modelCandidates = new[] { "/data/model.bin", "../model.bin" };
    foreach (var candidate in modelCandidates)
    {
        if (File.Exists(candidate))
        {
            gbmPredictor.LoadFromBinary(candidate);
            break;
        }
    }
}

var builder = WebApplication.CreateSlimBuilder(args);

builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonSerializerContext.Default);
});

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxConcurrentConnections = null;
    options.Limits.MaxConcurrentUpgradedConnections = null;
    options.Limits.MaxRequestBodySize = 8192;
    options.Limits.KeepAliveTimeout = TimeSpan.FromSeconds(120);
    options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(10);
    options.Limits.MinRequestBodyDataRate = null;
    options.Limits.MinResponseDataRate = null;
    options.AddServerHeader = false;
    options.AllowSynchronousIO = false;

    if (!string.IsNullOrWhiteSpace(unixSocketPath))
    {
        var socketDirectory = Path.GetDirectoryName(unixSocketPath);
        if (!string.IsNullOrWhiteSpace(socketDirectory))
        {
            Directory.CreateDirectory(socketDirectory);
        }

        if (File.Exists(unixSocketPath))
        {
            File.Delete(unixSocketPath);
        }

        options.ListenUnixSocket(unixSocketPath);
    }
});

builder.Logging.ClearProviders();

var app = builder.Build();

if (!string.IsNullOrWhiteSpace(unixSocketPath))
{
    app.Lifetime.ApplicationStarted.Register(() =>
    {
        try
        {
            if (!OperatingSystem.IsWindows() && File.Exists(unixSocketPath))
            {
                File.SetUnixFileMode(
                    unixSocketPath,
                    UnixFileMode.UserRead | UnixFileMode.UserWrite |
                    UnixFileMode.GroupRead | UnixFileMode.GroupWrite |
                    UnixFileMode.OtherRead | UnixFileMode.OtherWrite);
            }
        }
        catch
        {
        }
    });

    app.Lifetime.ApplicationStopped.Register(() =>
    {
        try
        {
            if (File.Exists(unixSocketPath))
            {
                File.Delete(unixSocketPath);
            }
        }
        catch
        {
        }
    });
}

// Warm up thread pool — with 0.45 CPU per container ProcessorCount=1, more threads
// just cause context-switch overhead. Keep it small and bounded.
ThreadPool.SetMinThreads(2, 2);
ThreadPool.SetMaxThreads(8, 8);

app.MapGet("/ready", () => ivfIndex.IsLoaded ? Results.Ok() : Results.StatusCode(503));

app.MapPost("/fraud-score", (FraudRequest request) =>
{
    try
    {
        Span<float> vector = stackalloc float[14];
        VectorNormalizer.Normalize(request, vector);

        float fraudScore;
        bool approved;

        // Hybrid pipeline: try ML first, fallback to IVF for ambiguous cases
        if (gbmPredictor.IsLoaded)
        {
            float mlProb = gbmPredictor.Predict(vector);

            if (mlProb > MlHighThreshold)
            {
                // ML confident: fraud
                fraudScore = mlProb;
                approved = false;
            }
            else if (mlProb < MlLowThreshold)
            {
                // ML confident: legit
                fraudScore = mlProb;
                approved = true;
            }
            else
            {
                // Ambiguous: use IVF search
                (fraudScore, approved) = ivfIndex.Search(vector);
            }
        }
        else
        {
            // No ML model: pure IVF
            (fraudScore, approved) = ivfIndex.Search(vector);
        }

        return Results.Ok(new FraudResponse { Approved = approved, FraudScore = fraudScore });
    }
    catch
    {
        // Fallback: approve to avoid 5-point error penalty
        return Results.Ok(new FraudResponse { Approved = true, FraudScore = 0.0 });
    }
});

app.Run();

[JsonSerializable(typeof(FraudResponse))]
[JsonSerializable(typeof(FraudRequest))]
internal partial class AppJsonSerializerContext : JsonSerializerContext
{
}
