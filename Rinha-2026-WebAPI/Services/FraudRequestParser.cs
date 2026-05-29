using System.Buffers.Text;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace Rinha_2026_WebAPI.Services;

/// <summary>
/// Stack-only value type that mirrors the fields actually consumed by
/// <see cref="VectorNormalizer"/>. Lives on the stack inside the request
/// handler and never escapes, so the parser is fully alloc-free.
/// </summary>
public struct FraudData
{
    // transaction
    public double Amount;
    public int Installments;
    public DateTime RequestedAt;

    // customer
    public double CustAvgAmount;
    public int TxCount24h;

    // merchant
    public int Mcc;
    public double MerchantAvgAmount;

    // terminal
    public bool IsOnline;
    public bool CardPresent;
    public double KmFromHome;

    // last_transaction (nullable)
    public bool HasLastTx;
    public DateTime LastTimestamp;
    public double KmFromCurrent;

    // computed: true when merchant.id is NOT in customer.known_merchants
    public bool UnknownMerchant;
}

/// <summary>
/// Hand-rolled <see cref="Utf8JsonReader"/> walker for the /fraud-score body.
/// Avoids the per-request allocations of the source-gen object binder
/// (FraudRequest + 4 nested objects + string[] + 4 strings) which were
/// the dominant GC pressure on the hot path.
/// </summary>
internal static class FraudRequestParser
{
    // Cap on customer.known_merchants tracked for the membership test.
    // The reference payloads never exceed ~16; 64 is a safe upper bound.
    private const int MaxKnownMerchants = 64;

    // Reusable per-thread buffers for known_merchants byte ranges.
    // Avoids stackalloc-scope issues with ref params and prevents per-request alloc.
    [ThreadStatic] private static int[]? t_kmStart;
    [ThreadStatic] private static int[]? t_kmLen;

    public static bool TryParse(ReadOnlySpan<byte> body, out FraudData data)
    {
        data = default;
        data.UnknownMerchant = true; // default: assume unknown until proven otherwise

        // Track the byte ranges (inside body) of merchant.id and each
        // known_merchants entry so the membership check works regardless of
        // field order in the JSON.
        var kmStart = t_kmStart ??= new int[MaxKnownMerchants];
        var kmLen = t_kmLen ??= new int[MaxKnownMerchants];
        int kmCount = 0;
        int midStart = -1, midLen = 0;

        var r = new Utf8JsonReader(body, isFinalBlock: true, state: default);
        if (!r.Read() || r.TokenType != JsonTokenType.StartObject) return false;

        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) break;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("transaction"u8))
            {
                r.Read(); // StartObject
                ParseTransaction(ref r, ref data);
            }
            else if (r.ValueTextEquals("customer"u8))
            {
                r.Read(); // StartObject
                ParseCustomer(ref r, ref data, body, kmStart, kmLen, ref kmCount);
            }
            else if (r.ValueTextEquals("merchant"u8))
            {
                r.Read(); // StartObject
                ParseMerchant(ref r, ref data, ref midStart, ref midLen);
            }
            else if (r.ValueTextEquals("terminal"u8))
            {
                r.Read(); // StartObject
                ParseTerminal(ref r, ref data);
            }
            else if (r.ValueTextEquals("last_transaction"u8))
            {
                r.Read();
                if (r.TokenType == JsonTokenType.Null)
                {
                    data.HasLastTx = false;
                }
                else if (r.TokenType == JsonTokenType.StartObject)
                {
                    data.HasLastTx = true;
                    ParseLastTx(ref r, ref data);
                }
            }
            else
            {
                r.Skip();
            }
        }

        // Resolve known_merchant membership using the captured byte ranges.
        if (midStart >= 0 && kmCount > 0)
        {
            var midSpan = body.Slice(midStart, midLen);
            for (int i = 0; i < kmCount; i++)
            {
                if (body.Slice(kmStart[i], kmLen[i]).SequenceEqual(midSpan))
                {
                    data.UnknownMerchant = false;
                    break;
                }
            }
        }

        return true;
    }

    private static void ParseTransaction(ref Utf8JsonReader r, ref FraudData d)
    {
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("amount"u8))
            {
                r.Read(); d.Amount = r.GetDouble();
            }
            else if (r.ValueTextEquals("installments"u8))
            {
                r.Read(); d.Installments = r.GetInt32();
            }
            else if (r.ValueTextEquals("requested_at"u8))
            {
                r.Read(); d.RequestedAt = r.GetDateTime();
            }
            else
            {
                r.Skip();
            }
        }
    }

    private static void ParseCustomer(
        ref Utf8JsonReader r, ref FraudData d, ReadOnlySpan<byte> body,
        int[] kmStart, int[] kmLen, ref int kmCount)
    {
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("avg_amount"u8))
            {
                r.Read(); d.CustAvgAmount = r.GetDouble();
            }
            else if (r.ValueTextEquals("tx_count_24h"u8))
            {
                r.Read(); d.TxCount24h = r.GetInt32();
            }
            else if (r.ValueTextEquals("known_merchants"u8))
            {
                r.Read(); // StartArray
                if (r.TokenType != JsonTokenType.StartArray) { r.Skip(); continue; }
                while (r.Read())
                {
                    if (r.TokenType == JsonTokenType.EndArray) break;
                    if (r.TokenType != JsonTokenType.String) continue;
                    if (kmCount >= MaxKnownMerchants) continue;
                    // Token starts at the opening quote; content starts +1.
                    // ValueSpan length is the unescaped byte length, which
                    // matches the source length when the string has no
                    // escapes (the case for merchant ids).
                    int start = (int)r.TokenStartIndex + 1;
                    int len = r.ValueSpan.Length;
                    kmStart[kmCount] = start;
                    kmLen[kmCount] = len;
                    kmCount++;
                }
            }
            else
            {
                r.Skip();
            }
        }
    }

    private static void ParseMerchant(
        ref Utf8JsonReader r, ref FraudData d, ref int midStart, ref int midLen)
    {
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("id"u8))
            {
                r.Read();
                midStart = (int)r.TokenStartIndex + 1;
                midLen = r.ValueSpan.Length;
            }
            else if (r.ValueTextEquals("mcc"u8))
            {
                r.Read();
                // MCC is a 4-digit numeric string. Utf8Parser handles leading
                // zeros and short tokens with no allocation.
                Utf8Parser.TryParse(r.ValueSpan, out d.Mcc, out _);
            }
            else if (r.ValueTextEquals("avg_amount"u8))
            {
                r.Read(); d.MerchantAvgAmount = r.GetDouble();
            }
            else
            {
                r.Skip();
            }
        }
    }

    private static void ParseTerminal(ref Utf8JsonReader r, ref FraudData d)
    {
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("is_online"u8))
            {
                r.Read(); d.IsOnline = r.GetBoolean();
            }
            else if (r.ValueTextEquals("card_present"u8))
            {
                r.Read(); d.CardPresent = r.GetBoolean();
            }
            else if (r.ValueTextEquals("km_from_home"u8))
            {
                r.Read(); d.KmFromHome = r.GetDouble();
            }
            else
            {
                r.Skip();
            }
        }
    }

    private static void ParseLastTx(ref Utf8JsonReader r, ref FraudData d)
    {
        while (r.Read())
        {
            if (r.TokenType == JsonTokenType.EndObject) return;
            if (r.TokenType != JsonTokenType.PropertyName) continue;

            if (r.ValueTextEquals("timestamp"u8))
            {
                r.Read(); d.LastTimestamp = r.GetDateTime();
            }
            else if (r.ValueTextEquals("km_from_current"u8))
            {
                r.Read(); d.KmFromCurrent = r.GetDouble();
            }
            else
            {
                r.Skip();
            }
        }
    }

    /// <summary>
    /// Writes a canonical fraud response (<c>{"approved":bool,"fraud_score":num}</c>)
    /// directly into the supplied buffer using UTF-8 formatters. Returns the
    /// number of bytes written. Buffer must be at least 64 bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WriteResponse(Span<byte> dest, bool approved, float fraudScore)
    {
        int w = 0;
        "{\"approved\":"u8.CopyTo(dest); w += "{\"approved\":"u8.Length;
        var b = approved ? "true"u8 : "false"u8;
        b.CopyTo(dest[w..]); w += b.Length;
        ",\"fraud_score\":"u8.CopyTo(dest[w..]); w += ",\"fraud_score\":"u8.Length;
        // 'G' gives the shortest round-trippable representation (e.g. 0.6
        // instead of 0.6000), matching the contract example in the README.
        Utf8Formatter.TryFormat(fraudScore, dest[w..], out int n,
            new System.Buffers.StandardFormat('G'));
        w += n;
        dest[w++] = (byte)'}';
        return w;
    }
}
