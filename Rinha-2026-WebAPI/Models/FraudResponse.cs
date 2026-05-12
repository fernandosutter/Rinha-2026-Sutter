using System.Text.Json.Serialization;

namespace Rinha_2026_WebAPI.Models;

public struct FraudResponse
{
    [JsonPropertyName("approved")]
    public bool Approved { get; set; }

    [JsonPropertyName("fraud_score")]
    public double FraudScore { get; set; }
}
