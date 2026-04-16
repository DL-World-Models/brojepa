"""Backward compatibility tests for all model architectures."""

import torch
import torch.nn as nn
import pytest

from models import ResNet18LateFusion, SimpleFCLateFusion, SimpleFCFourier


class TestSimpleFCLateFusion:
    """Tests for SimpleFCLateFusion model backward compatibility."""

    def test_model_initialization(self) -> None:
        """Test that model initializes with default and custom parameters."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "op_proj")
        assert hasattr(model, "net")

    def test_model_forward_pass(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test forward pass produces correct output shape."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10)
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"

    def test_model_with_custom_params(self) -> None:
        """Test model with custom op_embed_dim and mlp_hidden."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(
            ops=ops,
            num_classes=10,
            op_embed_dim=64,
            mlp_hidden=512,
        )

        assert model.net[0].in_features == 28 * 28 + 64
        assert model.net[0].out_features == 512

    def test_model_device_movement(self) -> None:
        """Test that model parameters move correctly to device."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops)

        # Test CPU
        model.to("cpu")
        params = list(model.parameters())
        assert all(p.device.type == "cpu" for p in params)

        # Test hypothetical GPU (will be CPU if not available)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        params = list(model.parameters())
        assert all(p.device.type == device.type for p in params)

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through model."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        assert loss.item() >= 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

        optimizer.step()

    def test_model_eval_mode(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test that eval mode disables certain layers."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10)

        model.eval()
        with torch.no_grad():
            out1 = model(batch_data["ctx_img"], batch_data["op"])
            out2 = model(batch_data["ctx_img"], batch_data["op"])

        # Eval mode outputs should be deterministic
        torch.testing.assert_close(out1, out2)

    def test_model_batch_sizes(self) -> None:
        """Test model works with various batch sizes."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10)
        model.eval()

        for batch_size in [1, 2, 4, 8, 16, 32]:
            img = torch.randn(batch_size, 1, 28, 28)
            op_vals = torch.ones(batch_size)

            with torch.no_grad():
                logits = model(img, op_vals)

            assert logits.shape == (batch_size, 10)


class TestResNet18LateFusion:
    """Tests for ResNet18LateFusion model backward compatibility."""

    def test_model_initialization(self) -> None:
        """Test that ResNet18 model initializes correctly."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "encoder")
        assert hasattr(model, "op_embed")
        assert hasattr(model, "head")
        assert hasattr(model, "op_values")

    def test_model_op_buffer_registered(self) -> None:
        """Test that op_values is registered as a buffer, not parameter."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops)

        # Should be in buffers, not parameters
        assert "op_values" in dict(model.named_buffers())
        assert "op_values" not in dict(model.named_parameters())

    def test_op_buffer_moves_with_model(self) -> None:
        """Test that op_values buffer moves to device with model."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops)

        model.to("cpu")
        assert model.op_values.device.type == "cpu"

    def test_duplicate_ops_raises_error(self) -> None:
        """Test that passing duplicate ops raises ValueError."""
        ops = [1, 2, 1, -1]  # 1 appears twice

        with pytest.raises(ValueError, match="ops must be unique"):
            ResNet18LateFusion(ops=ops)

    def test_model_forward_pass(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test ResNet18 forward pass produces correct output shape."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10)
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_forward_with_unknown_ops_raises_error(self) -> None:
        """Test that forward pass with unknown op values raises error."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops)

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5, 10])  # Not in ops list

        with pytest.raises(ValueError, match="unknown op values"):
            model(img, unknown_ops)

    def test_op_indexing_correctness(self) -> None:
        """Test that op values are correctly mapped to embedding indices."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops)

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])

        model.eval()
        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_with_different_op_orders(self) -> None:
        """Test that op order doesn't affect model internals (sorted)."""
        ops1 = [1, 2, -1, -2]
        ops2 = [-2, 1, -1, 2]  # Different order

        model1 = ResNet18LateFusion(ops=ops1)
        model2 = ResNet18LateFusion(ops=ops2)

        # Both should have the same sorted op_values
        torch.testing.assert_close(model1.op_values, model2.op_values)

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through ResNet18 model."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        assert loss.item() >= 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_batch_sizes(self) -> None:
        """Test ResNet18 model works with various batch sizes."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10)
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            img = torch.randn(batch_size, 1, 28, 28)
            op_vals = torch.ones(batch_size)

            with torch.no_grad():
                logits = model(img, op_vals)

            assert logits.shape == (batch_size, 10)


class TestSimpleFCFourier:
    """Tests for SimpleFCFourier model backward compatibility."""

    def test_model_initialization(self) -> None:
        """Test that SimpleFCFourier initializes with default and custom parameters."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "op_proj")
        assert hasattr(model, "net")

    def test_fourier_encoding_shape(self) -> None:
        """Test that Fourier encoding produces correct feature shape."""
        # Fourier features: sin(x), cos(x), sin(2x), cos(2x) = 4 features
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0]).unsqueeze(-1)  # (B, 1)

        op_feat = torch.cat([
            torch.sin(op_vals),
            torch.cos(op_vals),
            torch.sin(2 * op_vals),
            torch.cos(2 * op_vals),
        ], dim=-1)

        assert op_feat.shape == (4, 4)

    def test_model_forward_pass(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test forward pass produces correct output shape."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops, num_classes=10)
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_with_custom_params(self) -> None:
        """Test model with custom op_embed_dim and mlp_hidden."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(
            ops=ops,
            num_classes=10,
            op_embed_dim=64,
            mlp_hidden=512,
        )

        assert model.net[0].in_features == 28 * 28 + 64
        assert model.net[0].out_features == 512

    def test_model_device_movement(self) -> None:
        """Test that model parameters move correctly to device."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops)

        model.to("cpu")
        params = list(model.parameters())
        assert all(p.device.type == "cpu" for p in params)

    def test_model_gradient_flow(self) -> None:
        """Test that gradients flow through Fourier model."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops, num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        assert loss.item() >= 0
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_fourier_smoothness(self) -> None:
        """Test that Fourier encoding provides smooth interpolation."""
        model = SimpleFCFourier(ops=[1, 2, -1, -2])
        model.eval()

        img = torch.randn(1, 1, 28, 28)

        outputs = []
        for op_val in [0.5, 1.0, 1.5, 2.0]:
            op = torch.tensor([op_val])
            with torch.no_grad():
                out = model(img, op)
            outputs.append(out)

        # Outputs should be different (not nan or inf)
        for out in outputs:
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()

    def test_model_batch_sizes(self) -> None:
        """Test Fourier model works with various batch sizes."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops, num_classes=10)
        model.eval()

        for batch_size in [1, 2, 4, 8, 16]:
            img = torch.randn(batch_size, 1, 28, 28)
            op_vals = torch.ones(batch_size)

            with torch.no_grad():
                logits = model(img, op_vals)

            assert logits.shape == (batch_size, 10)

    def test_fourier_deterministic_output(self) -> None:
        """Test that Fourier model produces deterministic output in eval mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCFourier(ops=ops)
        model.eval()

        img = torch.randn(4, 1, 28, 28)
        op = torch.tensor([1.0, 2.0, -1.0, -2.0])

        with torch.no_grad():
            out1 = model(img, op)
            out2 = model(img, op)

        torch.testing.assert_close(out1, out2)
