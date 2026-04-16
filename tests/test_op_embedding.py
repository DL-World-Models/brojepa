"""Tests for operation embedding abstraction module."""

import torch
import torch.nn as nn
import pytest
from torch.nn import functional as F

from models import (
    OperationEmbedding,
    OperationEmbeddingLookup,
    OperationEmbeddingLinear,
    OperationEmbeddingFourier,
    create_operation_embedding,
)


class TestOperationEmbeddingLookup:
    """Tests for OperationEmbeddingLookup class."""

    def test_initialization(self) -> None:
        """Test OperationEmbeddingLookup initializes with correct embedding dimension."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)

        assert isinstance(embed, OperationEmbedding)
        assert embed.op_embed_dim == op_embed_dim
        assert len(embed.op_values) == len(ops)

    def test_forward_pass_single_op(self) -> None:
        """Test forward pass with single operation value."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)
        embed.eval()

        op_val = torch.tensor([1.0])
        with torch.no_grad():
            output = embed(op_val)

        assert output.shape == (1, op_embed_dim)
        assert not torch.isnan(output).any()

    def test_forward_pass_batch(self) -> None:
        """Test forward pass with batch of operations."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)
        embed.eval()

        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        with torch.no_grad():
            output = embed(op_vals)

        assert output.shape == (4, op_embed_dim)
        assert not torch.isnan(output).any()

    def test_different_batch_sizes(self) -> None:
        """Test OperationEmbeddingLookup with different batch sizes."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)
        embed.eval()

        for batch_size in [1, 2, 4, 8, 16]:
            op_vals = torch.ones(batch_size)
            with torch.no_grad():
                output = embed(op_vals)
            assert output.shape == (batch_size, op_embed_dim)

    def test_unknown_operation_raises_error(self) -> None:
        """Test that unknown operation values raise an error."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)

        op_val = torch.tensor([999.0])  # Unknown operation
        with pytest.raises(ValueError, match="unknown op values"):
            embed(op_val)

    def test_embeddings_are_unique(self) -> None:
        """Test that different operations have different embeddings."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)
        embed.eval()

        with torch.no_grad():
            embed1 = embed(torch.tensor([1.0]))
            embed2 = embed(torch.tensor([2.0]))

        # Embeddings should be different
        assert not torch.allclose(embed1, embed2)

    def test_device_movement(self) -> None:
        """Test that embeddings move to correct device."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)

        embed.to("cpu")
        params = list(embed.parameters())
        assert all(p.device.type == "cpu" for p in params)

    def test_gradient_flow(self) -> None:
        """Test that gradients can flow through the embedding."""
        ops = [1, 2, -1, -2]
        op_embed_dim = 32
        embed = OperationEmbeddingLookup(ops, op_embed_dim)

        op_val = torch.tensor([1.0], requires_grad=False)
        output = embed(op_val)
        loss = output.sum()
        loss.backward()

        # Check that embedding table has gradients
        assert embed.op_embed.weight.grad is not None


class TestOperationEmbeddingLinear:
    """Tests for OperationEmbeddingLinear class."""

    def test_initialization(self) -> None:
        """Test OperationEmbeddingLinear initializes correctly."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)

        assert isinstance(embed, OperationEmbedding)
        assert embed.op_embed_dim == op_embed_dim
        # Should have 2 linear layers and 1 ReLU in the projection MLP
        assert len(embed.proj) == 3

    def test_forward_pass_single_op(self) -> None:
        """Test forward pass with single operation value."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)
        embed.eval()

        op_val = torch.tensor([1.0])  # (B,) format
        with torch.no_grad():
            output = embed(op_val)

        assert output.shape == (1, op_embed_dim)
        assert not torch.isnan(output).any()

    def test_forward_pass_batch(self) -> None:
        """Test forward pass with batch of scalar operations."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)
        embed.eval()

        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        with torch.no_grad():
            output = embed(op_vals)

        assert output.shape == (4, op_embed_dim)

    def test_deterministic_output(self) -> None:
        """Test that neural network mapping is deterministic in eval mode."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)
        embed.eval()

        op_vals = torch.tensor([1.0, 2.0])
        with torch.no_grad():
            out1 = embed(op_vals)
            out2 = embed(op_vals)

        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through linear embedding."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)

        op_vals = torch.tensor([1.0, 2.0], requires_grad=True)
        output = embed(op_vals)
        loss = output.sum()
        loss.backward()

        # Check that linear layer has gradients
        for param in embed.proj.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_nonlinear_mapping(self) -> None:
        """Test that the linear embedding produces ReLU nonlinearity."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)
        embed.eval()

        # Different inputs should generally produce different outputs
        op_vals_a = torch.tensor([1.0])
        op_vals_b = torch.tensor([2.0])

        with torch.no_grad():
            out_a = embed(op_vals_a)
            out_b = embed(op_vals_b)

        # Outputs should be different
        assert not torch.allclose(out_a, out_b)

    def test_various_batch_sizes(self) -> None:
        """Test with various batch sizes."""
        op_embed_dim = 32
        embed = OperationEmbeddingLinear(op_embed_dim)
        embed.eval()

        for batch_size in [1, 2, 4, 8]:
            op_vals = torch.randn(batch_size)
            with torch.no_grad():
                output = embed(op_vals)
            assert output.shape == (batch_size, op_embed_dim)


class TestOperationEmbeddingFourier:
    """Tests for OperationEmbeddingFourier class."""

    def test_initialization(self) -> None:
        """Test OperationEmbeddingFourier initializes correctly."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)

        assert isinstance(embed, OperationEmbedding)
        assert embed.op_embed_dim == op_embed_dim

    def test_forward_pass_single_op(self) -> None:
        """Test forward pass with single operation value."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        op_val = torch.tensor([1.0])
        with torch.no_grad():
            output = embed(op_val)

        assert output.shape == (1, op_embed_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_pass_batch(self) -> None:
        """Test forward pass with batch."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        with torch.no_grad():
            output = embed(op_vals)

        assert output.shape == (4, op_embed_dim)

    def test_fourier_features_computation(self) -> None:
        """Test that Fourier features are computed correctly."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        op_val = torch.tensor([1.0])
        with torch.no_grad():
            output = embed(op_val)

        # Output should have meaningful values
        assert output.abs().max() > 0

    def test_smooth_interpolation(self) -> None:
        """Test that Fourier embedding provides smooth interpolation."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        # Nearby values should produce similar embeddings
        op_val_1 = torch.tensor([1.0])
        op_val_1_1 = torch.tensor([1.1])

        with torch.no_grad():
            out_1 = embed(op_val_1)
            out_1_1 = embed(op_val_1_1)

        # Should be similar but not identical
        similarity = F.cosine_similarity(out_1, out_1_1)
        assert 0.8 < similarity.item() < 1.0

    def test_periodic_behavior(self) -> None:
        """Test that Fourier encoding shows periodic behavior for sine/cosine components."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        # sin(x) and sin(x + 2π) should be similar
        op_val_x = torch.tensor([1.0])
        op_val_x_2pi = torch.tensor([1.0 + 2 * 3.14159265])

        with torch.no_grad():
            out_x = embed(op_val_x)
            out_x_2pi = embed(op_val_x_2pi)

        # Should be reasonably similar (both capture sine/cosine patterns)
        similarity = F.cosine_similarity(out_x, out_x_2pi)
        assert similarity.item() > 0.7

    def test_deterministic_output(self) -> None:
        """Test deterministic output in eval mode."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        op_vals = torch.tensor([1.0, 2.0])
        with torch.no_grad():
            out1 = embed(op_vals)
            out2 = embed(op_vals)

        torch.testing.assert_close(out1, out2)

    def test_gradient_flow(self) -> None:
        """Test that gradients can flow (through projection layer)."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)

        op_vals = torch.tensor([1.0, 2.0], requires_grad=True)
        output = embed(op_vals)
        loss = output.sum()
        loss.backward()

        # Check that projection layer has gradients
        for param in embed.proj.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_various_batch_sizes(self) -> None:
        """Test with various batch sizes."""
        op_embed_dim = 32
        embed = OperationEmbeddingFourier(op_embed_dim)
        embed.eval()

        for batch_size in [1, 2, 4, 8]:
            op_vals = torch.randn(batch_size)
            with torch.no_grad():
                output = embed(op_vals)
            assert output.shape == (batch_size, op_embed_dim)


class TestCreateOperationEmbedding:
    """Tests for the factory function create_operation_embedding."""

    def test_create_embedding_type(self) -> None:
        """Test factory creates OperationEmbeddingLookup for 'embedding' type."""
        ops = [1, 2, -1, -2]
        embed = create_operation_embedding("embedding", 32, ops)

        assert isinstance(embed, OperationEmbeddingLookup)

    def test_create_linear_type(self) -> None:
        """Test factory creates OperationEmbeddingLinear for 'linear' type."""
        embed = create_operation_embedding("linear", 32)

        assert isinstance(embed, OperationEmbeddingLinear)

    def test_create_fourier_type(self) -> None:
        """Test factory creates OperationEmbeddingFourier for 'fourier' type."""
        embed = create_operation_embedding("fourier", 32)

        assert isinstance(embed, OperationEmbeddingFourier)

    def test_invalid_type_raises_error(self) -> None:
        """Test that invalid embedding type raises ValueError."""
        with pytest.raises(ValueError, match="embedding_type must be"):
            create_operation_embedding("invalid_type", 32)  # type: ignore

    def test_embedding_requires_ops(self) -> None:
        """Test that 'embedding' type requires ops parameter."""
        with pytest.raises(ValueError, match="ops required"):
            create_operation_embedding("embedding", 32, ops=None)

    def test_linear_ignores_ops(self) -> None:
        """Test that 'linear' type works regardless of ops parameter."""
        # Should work without ops
        embed1 = create_operation_embedding("linear", 32)
        assert isinstance(embed1, OperationEmbeddingLinear)

        # Should still work if ops provided (ignored)
        embed2 = create_operation_embedding("linear", 32, [1, 2, -1, -2])
        assert isinstance(embed2, OperationEmbeddingLinear)

    def test_all_types_support_forward_pass(self) -> None:
        """Test that all created embeddings support forward pass."""
        ops = [1, 2, -1, -2]
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])

        for embedding_type in ["embedding", "linear", "fourier"]:
            ops_param = ops if embedding_type == "embedding" else None
            embed = create_operation_embedding(embedding_type, 32, ops_param)
            embed.eval()

            with torch.no_grad():
                output = embed(op_vals)

            assert output.shape == (4, 32)
            assert not torch.isnan(output).any()

    def test_embedding_dim_respected(self) -> None:
        """Test that specified embedding dimension is respected."""
        for embed_dim in [16, 32, 64]:
            for embedding_type in ["embedding", "linear", "fourier"]:
                ops_param = [1, 2, -1, -2] if embedding_type == "embedding" else None
                embed = create_operation_embedding(embedding_type, embed_dim, ops_param)

                assert embed.op_embed_dim == embed_dim

    def test_deterministic_factory_output(self) -> None:
        """Test that factory function produces working embeddings."""
        ops = [1, 2, -1, -2]
        op_vals = torch.tensor([1.0, 2.0])

        for embedding_type in ["embedding", "linear", "fourier"]:
            ops_param = ops if embedding_type == "embedding" else None
            embed = create_operation_embedding(embedding_type, 32, ops_param)
            embed.eval()

            with torch.no_grad():
                out1 = embed(op_vals)
                out2 = embed(op_vals)

            torch.testing.assert_close(out1, out2)
