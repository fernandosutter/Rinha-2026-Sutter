using System.Buffers;
using System.IO.Pipelines;
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

// Warm up thread pool — keep generous because Kestrel/IO and the work item
// for /fraud-score share the pool; capping low caused queueing on bursts.
ThreadPool.SetMinThreads(16, 16);

app.MapGet("/ready", () => ivfIndex.IsLoaded ? Results.Ok() : Results.StatusCode(503));

app.MapPost("/fraud-score", async (HttpContext ctx) =>
{
    // ----- 1. Drain request body into a contiguous span (alloc only if multi-segment) -----
    var bodyReader = ctx.Request.BodyReader;
    ReadResult readResult;
    while (true)
    {
        readResult = await bodyReader.ReadAsync();
        if (readResult.IsCompleted) break;
        // Not done yet: mark everything as "examined" so the pipe delivers more next round.
        bodyReader.AdvanceTo(readResult.Buffer.Start, readResult.Buffer.End);
    }

    var buffer = readResult.Buffer;
    int totalLen = (int)buffer.Length;
    byte[]? rented = null;
    ReadOnlySpan<byte> bodySpan;
    if (buffer.IsSingleSegment)
    {
        bodySpan = buffer.FirstSpan;
    }
    else
    {
        rented = ArrayPool<byte>.Shared.Rent(totalLen);
        buffer.CopyTo(rented);
        bodySpan = new ReadOnlySpan<byte>(rented, 0, totalLen);
    }

    float fraudScore;
    bool approved;
    try
    {
        if (!FraudRequestParser.TryParse(bodySpan, out var data))
        {
            // Malformed body: respond approve to avoid the 5-point error penalty.
            approved = true;
            fraudScore = 0f;
        }
        else
        {
            Span<float> vector = stackalloc float[14];
            VectorNormalizer.Normalize(data, vector);

            if (gbmPredictor.IsLoaded)
            {
                float mlProb = gbmPredictor.Predict(vector);
                if (mlProb > MlHighThreshold)
                {
                    fraudScore = mlProb; approved = false;
                }
                else if (mlProb < MlLowThreshold)
                {
                    fraudScore = mlProb; approved = true;
                }
                else
                {
                    (fraudScore, approved) = ivfIndex.Search(vector);
                }
            }
            else
            {
                (fraudScore, approved) = ivfIndex.Search(vector);
            }
        }
    }
    catch
    {
        approved = true;
        fraudScore = 0f;
    }
    finally
    {
        bodyReader.AdvanceTo(buffer.End);
        if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
    }

    // ----- 2. Write canonical JSON response straight into the BodyWriter (zero-alloc) -----
    var resp = ctx.Response;
    resp.StatusCode = 200;
    resp.ContentType = "application/json";
    var writer = resp.BodyWriter;
    var outBuf = writer.GetSpan(64);
    int written = FraudRequestParser.WriteResponse(outBuf, approved, fraudScore);
    writer.Advance(written);
    await writer.FlushAsync();
});

app.Run();

[JsonSerializable(typeof(FraudResponse))]
[JsonSerializable(typeof(FraudRequest))]
internal partial class AppJsonSerializerContext : JsonSerializerContext
{
}
