using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Rinha_2026_WebAPI.Services;

public static class SimdDistance
{
    // 14 dimensions: process using Vector256 (8 floats) + Vector128 (4 floats) + 2 scalar
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EuclideanDistanceSquared(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (Avx2.IsSupported && a.Length >= 8)
        {
            // Process first 8 dims with AVX2
            var va = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(a));
            var vb = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(b));
            var diff = Avx.Subtract(va, vb);
            var sq = Avx.Multiply(diff, diff);

            // Process dims 8-11 with SSE
            var va2 = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(a), 8);
            var vb2 = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(b), 8);
            var diff2 = Sse.Subtract(va2, vb2);
            var sq2 = Sse.Multiply(diff2, diff2);

            // Remaining 2 dims scalar
            float d12 = a[12] - b[12];
            float d13 = a[13] - b[13];
            float scalarSum = d12 * d12 + d13 * d13;

            // Horizontal sum of AVX vector (8 floats)
            var sum256 = sq;
            var hi128 = Avx.ExtractVector128(sum256, 1);
            var lo128 = Avx.ExtractVector128(sum256, 0);
            var sum128 = Sse.Add(hi128, lo128);
            sum128 = Sse.Add(sum128, sq2);

            // Final horizontal sum of 4 floats
            var shuf = Sse.MoveHighToLow(sum128, sum128);
            var sums = Sse.Add(sum128, shuf);
            var hi = Sse.Shuffle(sums, sums, 0x01);
            var result = Sse.AddScalar(sums, hi);

            return result.ToScalar() + scalarSum;
        }

        // Fallback: manual unrolling
        float d0 = a[0] - b[0];
        float d1 = a[1] - b[1];
        float d2 = a[2] - b[2];
        float d3 = a[3] - b[3];
        float d4 = a[4] - b[4];
        float d5 = a[5] - b[5];
        float d6 = a[6] - b[6];
        float d7 = a[7] - b[7];
        float d8 = a[8] - b[8];
        float d9 = a[9] - b[9];
        float d10 = a[10] - b[10];
        float d11 = a[11] - b[11];
        float fd12 = a[12] - b[12];
        float fd13 = a[13] - b[13];

        return d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3
             + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7
             + d8 * d8 + d9 * d9 + d10 * d10 + d11 * d11
             + fd12 * fd12 + fd13 * fd13;
    }

    // Distance from float query to Half-stored vector (IVF vectors stored as Half).
    // Hot path: convert the 14 halves to floats once (Half->float is JIT-intrinsified
    // to F16C VCVTSH2SS on Haswell/Broadwell+, ~1c throughput each) then reuse the
    // already-vectorized float distance kernel above.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EuclideanDistanceSquaredHalf(ReadOnlySpan<float> query, ReadOnlySpan<Half> vec)
    {
        // 16 floats keeps the buffer 64-byte aligned in stack and lets the
        // AVX2 kernel above read the first 8/4/2 lanes without bounds checks.
        Span<float> buf = stackalloc float[16];
        ref Half src = ref MemoryMarshal.GetReference(vec);
        buf[0]  = (float)Unsafe.Add(ref src, 0);
        buf[1]  = (float)Unsafe.Add(ref src, 1);
        buf[2]  = (float)Unsafe.Add(ref src, 2);
        buf[3]  = (float)Unsafe.Add(ref src, 3);
        buf[4]  = (float)Unsafe.Add(ref src, 4);
        buf[5]  = (float)Unsafe.Add(ref src, 5);
        buf[6]  = (float)Unsafe.Add(ref src, 6);
        buf[7]  = (float)Unsafe.Add(ref src, 7);
        buf[8]  = (float)Unsafe.Add(ref src, 8);
        buf[9]  = (float)Unsafe.Add(ref src, 9);
        buf[10] = (float)Unsafe.Add(ref src, 10);
        buf[11] = (float)Unsafe.Add(ref src, 11);
        buf[12] = (float)Unsafe.Add(ref src, 12);
        buf[13] = (float)Unsafe.Add(ref src, 13);
        return EuclideanDistanceSquared(query, buf);
    }
}
