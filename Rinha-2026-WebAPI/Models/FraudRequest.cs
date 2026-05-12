using System.Text.Json.Serialization;

namespace Rinha_2026_WebAPI.Models;

public sealed class FraudRequest
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = default!;

    [JsonPropertyName("transaction")]
    public TransactionData Transaction { get; set; } = default!;

    [JsonPropertyName("customer")]
    public CustomerData Customer { get; set; } = default!;

    [JsonPropertyName("merchant")]
    public MerchantData Merchant { get; set; } = default!;

    [JsonPropertyName("terminal")]
    public TerminalData Terminal { get; set; } = default!;

    [JsonPropertyName("last_transaction")]
    public LastTransactionData? LastTransaction { get; set; }
}

public sealed class TransactionData
{
    [JsonPropertyName("amount")]
    public double Amount { get; set; }

    [JsonPropertyName("installments")]
    public int Installments { get; set; }

    [JsonPropertyName("requested_at")]
    public DateTime RequestedAt { get; set; }
}

public sealed class CustomerData
{
    [JsonPropertyName("avg_amount")]
    public double AvgAmount { get; set; }

    [JsonPropertyName("tx_count_24h")]
    public int TxCount24h { get; set; }

    [JsonPropertyName("known_merchants")]
    public string[] KnownMerchants { get; set; } = [];
}

public sealed class MerchantData
{
    [JsonPropertyName("id")]
    public string Id { get; set; } = default!;

    [JsonPropertyName("mcc")]
    public string Mcc { get; set; } = default!;

    [JsonPropertyName("avg_amount")]
    public double AvgAmount { get; set; }
}

public sealed class TerminalData
{
    [JsonPropertyName("is_online")]
    public bool IsOnline { get; set; }

    [JsonPropertyName("card_present")]
    public bool CardPresent { get; set; }

    [JsonPropertyName("km_from_home")]
    public double KmFromHome { get; set; }
}

public sealed class LastTransactionData
{
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }

    [JsonPropertyName("km_from_current")]
    public double KmFromCurrent { get; set; }
}
