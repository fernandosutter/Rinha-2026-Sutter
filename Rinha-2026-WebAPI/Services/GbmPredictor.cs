using System.Runtime.CompilerServices;

namespace Rinha_2026_WebAPI.Services;

/// <summary>
/// Binary GBM model inference. Reads a compact binary format exported by train_model.py.
/// Layout: [n_trees: int32] [tree_0] [tree_1] ...
/// Each tree: [n_nodes: int32] [node_0] [node_1] ...
/// Each node (13 bytes): [feature_index: byte] [threshold: float32] [left: int16] [right: int16] [leaf_value: float32]
/// Leaf nodes: feature_index == 0xFF, leaf_value contains the value.
/// </summary>
public sealed class GbmPredictor
{
    private const byte LeafMarker = 0xFF;

    // Flat arrays for all trees — cache friendly
    private byte[] _featureIndices = [];
    private float[] _thresholds = [];
    private short[] _leftChildren = [];
    private short[] _rightChildren = [];
    private float[] _leafValues = [];
    private int[] _treeOffsets = []; // start offset of each tree in the flat arrays
    private int _nTrees;

    public bool IsLoaded => _nTrees > 0;

    public void LoadFromBinary(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        _nTrees = br.ReadInt32();

        // First pass: count total nodes
        long startPos = fs.Position;
        int totalNodes = 0;
        for (int t = 0; t < _nTrees; t++)
        {
            int nNodes = br.ReadInt32();
            totalNodes += nNodes;
            fs.Seek(nNodes * 13L, SeekOrigin.Current); // skip node data
        }

        // Allocate flat arrays
        _featureIndices = new byte[totalNodes];
        _thresholds = new float[totalNodes];
        _leftChildren = new short[totalNodes];
        _rightChildren = new short[totalNodes];
        _leafValues = new float[totalNodes];
        _treeOffsets = new int[_nTrees + 1];

        // Second pass: read nodes
        fs.Position = startPos;
        int offset = 0;
        for (int t = 0; t < _nTrees; t++)
        {
            _treeOffsets[t] = offset;
            int nNodes = br.ReadInt32();
            for (int n = 0; n < nNodes; n++)
            {
                _featureIndices[offset] = br.ReadByte();
                _thresholds[offset] = br.ReadSingle();
                _leftChildren[offset] = br.ReadInt16();
                _rightChildren[offset] = br.ReadInt16();
                _leafValues[offset] = br.ReadSingle();
                offset++;
            }
        }
        _treeOffsets[_nTrees] = offset;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Predict(ReadOnlySpan<float> features)
    {
        float sum = 0f;
        for (int t = 0; t < _nTrees; t++)
        {
            int baseOffset = _treeOffsets[t];
            int nodeIdx = 0;

            while (true)
            {
                int idx = baseOffset + nodeIdx;
                byte featureIdx = _featureIndices[idx];

                if (featureIdx == LeafMarker)
                {
                    sum += _leafValues[idx];
                    break;
                }

                if (features[featureIdx] <= _thresholds[idx])
                    nodeIdx = _leftChildren[idx];
                else
                    nodeIdx = _rightChildren[idx];
            }
        }

        return Sigmoid(sum);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x)
    {
        return 1f / (1f + MathF.Exp(-x));
    }
}
