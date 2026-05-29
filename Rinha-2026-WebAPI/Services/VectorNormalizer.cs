using System.Collections.Frozen;
using Rinha_2026_WebAPI.Models;

namespace Rinha_2026_WebAPI.Services;

public static class VectorNormalizer
{
    private const double MaxAmount = 10000.0;
    private const double MaxInstallments = 12.0;
    private const double AmountVsAvgRatio = 10.0;
    private const double MaxMinutes = 1440.0;
    private const double MaxKm = 1000.0;
    private const double MaxTxCount24h = 20.0;
    private const double MaxMerchantAvgAmount = 10000.0;

    private static readonly FrozenDictionary<string, float> MccRisk = new Dictionary<string, float>
    {
        ["5411"] = 0.15f,
        ["5812"] = 0.30f,
        ["5912"] = 0.20f,
        ["5944"] = 0.45f,
        ["7801"] = 0.80f,
        ["7802"] = 0.75f,
        ["7995"] = 0.85f,
        ["4511"] = 0.35f,
        ["5311"] = 0.25f,
        ["5999"] = 0.50f,
    }.ToFrozenDictionary();

    // Mirror of MccRisk keyed by parsed int – used by the alloc-free
    // FraudRequestParser path, which stores MCC as int to skip a string alloc.
    private static readonly FrozenDictionary<int, float> MccRiskInt = new Dictionary<int, float>
    {
        [5411] = 0.15f,
        [5812] = 0.30f,
        [5912] = 0.20f,
        [5944] = 0.45f,
        [7801] = 0.80f,
        [7802] = 0.75f,
        [7995] = 0.85f,
        [4511] = 0.35f,
        [5311] = 0.25f,
        [5999] = 0.50f,
    }.ToFrozenDictionary();

    public static void Normalize(FraudRequest req, Span<float> vector)
    {
        var tx = req.Transaction;
        var cust = req.Customer;
        var merch = req.Merchant;
        var term = req.Terminal;
        var last = req.LastTransaction;

        // 0: amount
        vector[0] = Clamp01((float)(tx.Amount / MaxAmount));

        // 1: installments
        vector[1] = Clamp01((float)(tx.Installments / MaxInstallments));

        // 2: amount_vs_avg
        vector[2] = Clamp01((float)(tx.Amount / cust.AvgAmount / AmountVsAvgRatio));

        // 3: hour_of_day
        vector[3] = tx.RequestedAt.Hour / 23f;

        // 4: day_of_week (Mon=0, Sun=6)
        int dow = ((int)tx.RequestedAt.DayOfWeek + 6) % 7; // Convert .NET (Sun=0) to Mon=0
        vector[4] = dow / 6f;

        // 5: minutes_since_last_tx
        if (last is null)
        {
            vector[5] = -1f;
        }
        else
        {
            double minutes = (tx.RequestedAt - last.Timestamp).TotalMinutes;
            vector[5] = Clamp01((float)(minutes / MaxMinutes));
        }

        // 6: km_from_last_tx
        if (last is null)
        {
            vector[6] = -1f;
        }
        else
        {
            vector[6] = Clamp01((float)(last.KmFromCurrent / MaxKm));
        }

        // 7: km_from_home
        vector[7] = Clamp01((float)(term.KmFromHome / MaxKm));

        // 8: tx_count_24h
        vector[8] = Clamp01((float)(cust.TxCount24h / MaxTxCount24h));

        // 9: is_online
        vector[9] = term.IsOnline ? 1f : 0f;

        // 10: card_present
        vector[10] = term.CardPresent ? 1f : 0f;

        // 11: unknown_merchant
        bool known = false;
        for (int i = 0; i < cust.KnownMerchants.Length; i++)
        {
            if (cust.KnownMerchants[i] == merch.Id)
            {
                known = true;
                break;
            }
        }
        vector[11] = known ? 0f : 1f;

        // 12: mcc_risk
        vector[12] = MccRisk.GetValueOrDefault(merch.Mcc, 0.5f);

        // 13: merchant_avg_amount
        vector[13] = Clamp01((float)(merch.AvgAmount / MaxMerchantAvgAmount));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float Clamp01(float value)
    {
        if (value < 0f) return 0f;
        if (value > 1f) return 1f;
        return value;
    }

    /// <summary>
    /// Alloc-free overload that consumes the manually-parsed <see cref="FraudData"/>
    /// instead of the source-generated <see cref="FraudRequest"/> object graph.
    /// Mirrors <see cref="Normalize(FraudRequest, Span{float})"/> field-for-field.
    /// </summary>
    public static void Normalize(in FraudData req, Span<float> vector)
    {
        // 0: amount
        vector[0] = Clamp01((float)(req.Amount / MaxAmount));

        // 1: installments
        vector[1] = Clamp01((float)(req.Installments / MaxInstallments));

        // 2: amount_vs_avg
        vector[2] = Clamp01((float)(req.Amount / req.CustAvgAmount / AmountVsAvgRatio));

        // 3: hour_of_day
        vector[3] = req.RequestedAt.Hour / 23f;

        // 4: day_of_week (Mon=0, Sun=6)
        int dow = ((int)req.RequestedAt.DayOfWeek + 6) % 7;
        vector[4] = dow / 6f;

        // 5: minutes_since_last_tx
        if (!req.HasLastTx)
        {
            vector[5] = -1f;
            vector[6] = -1f;
        }
        else
        {
            double minutes = (req.RequestedAt - req.LastTimestamp).TotalMinutes;
            vector[5] = Clamp01((float)(minutes / MaxMinutes));
            vector[6] = Clamp01((float)(req.KmFromCurrent / MaxKm));
        }

        // 7: km_from_home
        vector[7] = Clamp01((float)(req.KmFromHome / MaxKm));

        // 8: tx_count_24h
        vector[8] = Clamp01((float)(req.TxCount24h / MaxTxCount24h));

        // 9/10: terminal flags
        vector[9]  = req.IsOnline    ? 1f : 0f;
        vector[10] = req.CardPresent ? 1f : 0f;

        // 11: unknown_merchant – pre-computed by the parser
        vector[11] = req.UnknownMerchant ? 1f : 0f;

        // 12: mcc_risk – int-keyed lookup, no string alloc
        vector[12] = MccRiskInt.GetValueOrDefault(req.Mcc, 0.5f);

        // 13: merchant_avg_amount
        vector[13] = Clamp01((float)(req.MerchantAvgAmount / MaxMerchantAvgAmount));
    }
}
