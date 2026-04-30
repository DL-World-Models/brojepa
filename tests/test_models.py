"""Backward compatibility tests for all model architectures."""

import torch
import torch.nn as nn
import pytest

from models import ResNet18LateFusion, SimpleFCLateFusion
from models.jepa_model import JepaPredictor
from models.rotation import ActionRotation


class TestActionRotation:
    """Tests that baseline action rotation stays aligned with JEPA rotation."""

    @pytest.mark.parametrize("normalize", [False, True])
    def test_block_rotation_matches_jepa_predictor(self, normalize: bool) -> None:
        torch.manual_seed(0)
        latent_dim = 33
        action_rotation = ActionRotation(
            dim=latent_dim,
            rotation_type="block_rotation",
            init_mode="cyclic_harmonics",
            period=10,
            learnable=False,
            normalize=normalize,
        )
        jepa_predictor = JepaPredictor(
            latent_dim=latent_dim,
            predictor_type="block_rotation",
            rotation_init_mode="cyclic_harmonics",
            rotation_period=10,
            rotation_learnable=False,
            normalize=normalize,
        )
        x = torch.randn(5, latent_dim)
        actions = torch.tensor([1.0, 2.0, -3.0, 4.0, -5.0])

        action_out = action_rotation(x, actions)
        jepa_out = jepa_predictor(x, actions_embedded=torch.empty(5, 0), action_values=actions)

        torch.testing.assert_close(action_rotation.rotation_angles, jepa_predictor.rotation_angles)
        torch.testing.assert_close(action_out, jepa_out)

    @pytest.mark.parametrize("normalize", [False, True])
    def test_simple_rotation_matches_jepa_predictor(self, normalize: bool) -> None:
        torch.manual_seed(0)
        latent_dim = 33
        action_rotation = ActionRotation(
            dim=latent_dim,
            rotation_type="simple_rotation",
            angle_init=0.25,
            learnable=False,
            normalize=normalize,
        )
        jepa_predictor = JepaPredictor(
            latent_dim=latent_dim,
            predictor_type="simple_rotation",
            rotation_angle_init=0.25,
            rotation_learnable=False,
            normalize=normalize,
        )
        x = torch.randn(5, latent_dim)
        actions = torch.tensor([1.0, 2.0, -3.0, 4.0, -5.0])

        action_out = action_rotation(x, actions)
        jepa_out = jepa_predictor(x, actions_embedded=torch.empty(5, 0), action_values=actions)

        torch.testing.assert_close(action_rotation.rotation_angles, jepa_predictor.rotation_angles)
        torch.testing.assert_close(action_out, jepa_out)


class TestSimpleFCLateFusion:
    """Tests for SimpleFCLateFusion model backward compatibility."""

    def test_model_initialization(self) -> None:
        """Test that model initializes with default and custom parameters."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "op_embed")
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

    def test_model_with_pre_head_simple_rotation(self) -> None:
        """Test simple rotation applies to image features before the head."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(
            ops=ops,
            num_classes=10,
            predictor_type="simple_rotation",
            predictor_normalize=False,
            use_action_rotation=True,
            include_op_embedding_in_head=False,
        )
        model.eval()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)
        assert model.encoder[0].out_features == 256
        assert model.classifier.in_features == 256
        assert model.action_rotation.rotation_type == "simple_rotation"
        assert model.action_rotation.rotation_angles is not None

    def test_model_with_pre_head_block_rotation(self) -> None:
        """Test block rotation applies to image features before the head."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(
            ops=ops,
            num_classes=10,
            predictor_type="block_rotation",
            predictor_normalize=False,
            rotation_init_mode="cyclic_harmonics",
            rotation_learnable=False,
            use_action_rotation=True,
            include_op_embedding_in_head=False,
        )
        model.eval()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)
        assert model.encoder[0].out_features == 256
        assert model.classifier.in_features == 256
        assert model.action_rotation.rotation_type == "block_rotation"
        assert model.action_rotation.rotation_angles.shape == (128,)

    def test_model_with_deeper_split_encoder(self) -> None:
        """Test the rotation path can use multiple encoder layers."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(
            ops=ops,
            num_classes=10,
            simple_encoder_depth=3,
            predictor_type="block_rotation",
            use_action_rotation=True,
            include_op_embedding_in_head=False,
        )

        linear_layers = [layer for layer in model.encoder if isinstance(layer, nn.Linear)]

        assert len(linear_layers) == 3
        assert linear_layers[0].in_features == 28 * 28
        assert all(layer.out_features == 256 for layer in linear_layers)

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

    def test_model_initialization_embedding_mode(self) -> None:
        """Test SimpleFCLateFusion initialization with explicit embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, op_embedding_type="embedding")

        assert model.op_embedding_type == "embedding"

    def test_model_initialization_linear_mode(self) -> None:
        """Test SimpleFCLateFusion initialization with linear embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, op_embedding_type="linear")

        assert model.op_embedding_type == "linear"

    def test_model_initialization_fourier_mode(self) -> None:
        """Test SimpleFCLateFusion initialization with fourier embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, op_embedding_type="fourier")

        assert model.op_embedding_type == "fourier"

    def test_invalid_embedding_type_raises_error(self) -> None:
        """Test that invalid embedding type raises ValueError."""
        ops = [1, 2, -1, -2]

        with pytest.raises(ValueError, match="embedding_type must be"):
            SimpleFCLateFusion(ops=ops, op_embedding_type="invalid_type")  # type: ignore

    def test_duplicate_ops_embedding_mode_raises_error(self) -> None:
        """Test that duplicate ops raises error in embedding mode."""
        ops = [1, 2, 1, -1]  # 1 appears twice

        with pytest.raises(ValueError, match="ops must be unique"):
            SimpleFCLateFusion(ops=ops, op_embedding_type="embedding")

    def test_duplicate_ops_linear_mode_allowed(self) -> None:
        """Test that duplicate ops are allowed in linear mode."""
        ops = [1, 2, 1, -1]  # duplicates are fine for linear

        model = SimpleFCLateFusion(ops=ops, op_embedding_type="linear")
        assert model.op_embedding_type == "linear"

    def test_model_forward_pass_embedding(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test SimpleFCLateFusion forward pass with embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="embedding")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_forward_pass_linear(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test SimpleFCLateFusion forward pass with linear embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="linear")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_forward_pass_fourier(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test SimpleFCLateFusion forward pass with Fourier embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="fourier")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_forward_with_unknown_ops_embedding_raises_error(self) -> None:
        """Test that forward pass with unknown ops raises error in embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, op_embedding_type="embedding")

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5, 10])  # Not in ops list

        with pytest.raises(ValueError, match="unknown op values"):
            model(img, unknown_ops)

    def test_forward_with_unknown_ops_linear_allowed(self) -> None:
        """Test that forward pass with unknown ops works in linear mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, op_embedding_type="linear")
        model.eval()

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5.0, 10.0])  # Any values are ok for linear

        with torch.no_grad():
            logits = model(img, unknown_ops)

        assert logits.shape == (2, 10)

    def test_gradient_flow_embedding(self) -> None:
        """Test gradient flow with embedding mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="embedding")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_flow_linear(self) -> None:
        """Test gradient flow with linear mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="linear")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_gradient_flow_fourier(self) -> None:
        """Test gradient flow with fourier mode."""
        ops = [1, 2, -1, -2]
        model = SimpleFCLateFusion(ops=ops, num_classes=10, op_embedding_type="fourier")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])
        targets = torch.tensor([1, 3, 8, 5])

        logits = model(img, op_vals)
        loss = criterion(logits, targets)
        loss.backward()

        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestResNet18LateFusion:
    """Tests for ResNet18LateFusion model backward compatibility."""

    def test_model_initialization_default_embedding(self) -> None:
        """Test that ResNet18 model initializes with default embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10)

        assert isinstance(model, nn.Module)
        assert hasattr(model, "encoder")
        assert hasattr(model, "op_embed")
        assert hasattr(model, "head")
        assert model.op_embedding_type == "embedding"

    def test_model_initialization_embedding_mode(self) -> None:
        """Test ResNet18 initialization with explicit embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

        assert model.op_embedding_type == "embedding"
        # op_values is in the op_embed submodule now
        assert hasattr(model.op_embed, "op_values")

    def test_model_initialization_linear_mode(self) -> None:
        """Test ResNet18 initialization with linear embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="linear")

        assert model.op_embedding_type == "linear"

    def test_model_initialization_fourier_mode(self) -> None:
        """Test ResNet18 initialization with fourier embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="fourier")

        assert model.op_embedding_type == "fourier"

    def test_invalid_embedding_type_raises_error(self) -> None:
        """Test that invalid embedding type raises ValueError."""
        ops = [1, 2, -1, -2]

        with pytest.raises(ValueError, match="embedding_type must be"):
            ResNet18LateFusion(ops=ops, op_embedding_type="invalid_type")  # type: ignore

    def test_model_op_buffer_registered(self) -> None:
        """Test that op_values is registered as a buffer in embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

        # op_values is a buffer in the op_embed submodule
        assert "op_values" in dict(model.op_embed.named_buffers())
        assert "op_values" not in dict(model.op_embed.named_parameters())

    def test_op_buffer_moves_with_model(self) -> None:
        """Test that op_values buffer moves to device with model."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

        model.to("cpu")
        assert model.op_embed.op_values.device.type == "cpu"

    def test_duplicate_ops_embedding_mode_raises_error(self) -> None:
        """Test that duplicate ops raises error in embedding mode."""
        ops = [1, 2, 1, -1]  # 1 appears twice

        with pytest.raises(ValueError, match="ops must be unique"):
            ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

    def test_duplicate_ops_linear_mode_allowed(self) -> None:
        """Test that duplicate ops are allowed in linear mode."""
        ops = [1, 2, 1, -1]  # duplicates are fine for linear

        model = ResNet18LateFusion(ops=ops, op_embedding_type="linear")
        assert model.op_embedding_type == "linear"

    def test_model_forward_pass_embedding(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test ResNet18 forward pass with embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="embedding")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_forward_pass_linear(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test ResNet18 forward pass with linear embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="linear")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_forward_pass_fourier(self, batch_data: dict[str, torch.Tensor]) -> None:
        """Test ResNet18 forward pass with Fourier embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="fourier")
        model.eval()

        img = batch_data["ctx_img"]
        op_vals = batch_data["op"]

        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_forward_with_unknown_ops_embedding_raises_error(self) -> None:
        """Test that forward pass with unknown ops raises error in embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5, 10])  # Not in ops list

        with pytest.raises(ValueError, match="unknown op values"):
            model(img, unknown_ops)

    def test_forward_with_unknown_ops_linear_allowed(self) -> None:
        """Test that forward pass with unknown ops works in linear mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="linear")
        model.eval()

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5.0, 10.0])  # Any values are ok for linear

        with torch.no_grad():
            logits = model(img, unknown_ops)

        assert logits.shape == (2, 10)

    def test_forward_with_unknown_ops_fourier_allowed(self) -> None:
        """Test that forward pass with unknown ops works in Fourier mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="fourier")
        model.eval()

        img = torch.randn(2, 1, 28, 28)
        unknown_ops = torch.tensor([5.0, 10.0])  # Any values work with Fourier

        with torch.no_grad():
            logits = model(img, unknown_ops)

        assert logits.shape == (2, 10)

    def test_op_indexing_correctness_embedding(self) -> None:
        """Test that op values are correctly mapped in embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, op_embedding_type="embedding")

        img = torch.randn(4, 1, 28, 28)
        op_vals = torch.tensor([1.0, 2.0, -1.0, -2.0])

        model.eval()
        with torch.no_grad():
            logits = model(img, op_vals)

        assert logits.shape == (4, 10)

    def test_model_with_different_op_orders_embedding(self) -> None:
        """Test that op order doesn't matter in embedding mode (sorted)."""
        ops1 = [1, 2, -1, -2]
        ops2 = [-2, 1, -1, 2]

        model1 = ResNet18LateFusion(ops=ops1, op_embedding_type="embedding")
        model2 = ResNet18LateFusion(ops=ops2, op_embedding_type="embedding")

        # Both models should sort ops the same way
        torch.testing.assert_close(model1.op_embed.op_values, model2.op_embed.op_values)

    def test_model_gradient_flow_embedding(self) -> None:
        """Test that gradients flow through embedding mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="embedding")
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

    def test_model_gradient_flow_linear(self) -> None:
        """Test that gradients flow through linear mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="linear")
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

    def test_model_gradient_flow_fourier(self) -> None:
        """Test that gradients flow through Fourier mode."""
        ops = [1, 2, -1, -2]
        model = ResNet18LateFusion(ops=ops, num_classes=10, op_embedding_type="fourier")
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

    def test_model_batch_sizes_all_modes(self) -> None:
        """Test all embedding modes work with various batch sizes."""
        ops = [1, 2, -1, -2]

        for embedding_type in ["embedding", "linear", "fourier"]:
            model = ResNet18LateFusion(ops=ops, op_embedding_type=embedding_type)
            model.eval()

            for batch_size in [1, 2, 4, 8]:
                img = torch.randn(batch_size, 1, 28, 28)
                op_vals = torch.ones(batch_size)

                with torch.no_grad():
                    logits = model(img, op_vals)

                assert logits.shape == (batch_size, 10)
