"""Tests for promptingnemo.models.text_ctc_model."""

import pytest
import torch

try:
    from promptingnemo.models.text_ctc_model import (
        CharacterEmbedding,
        LearnedUpsampler,
        TextCTCTagger,
    )
    _HAS_LIGHTNING = True
except ImportError:
    _HAS_LIGHTNING = False

pytestmark = pytest.mark.skipif(not _HAS_LIGHTNING, reason="lightning not installed")


class TestCharacterEmbedding:
    def test_output_shape(self):
        emb = CharacterEmbedding(char_vocab_size=100, embed_dim=64, hidden_dim=128)
        char_ids = torch.randint(0, 100, (2, 20))
        out = emb(char_ids)
        assert out.shape == (2, 20, 128)

    def test_padding_handled(self):
        emb = CharacterEmbedding(char_vocab_size=100, embed_dim=64, hidden_dim=128)
        char_ids = torch.zeros(1, 10, dtype=torch.long)  # all padding
        out = emb(char_ids)
        assert out.shape == (1, 10, 128)


class TestLearnedUpsampler:
    def test_doubles_length(self):
        up = LearnedUpsampler(hidden_dim=128, factor=2)
        x = torch.randn(2, 10, 128)
        lengths = torch.tensor([10, 8])
        out, new_lengths = up(x, lengths)
        assert out.shape == (2, 20, 128)
        assert new_lengths.tolist() == [20, 16]

    def test_triples_length(self):
        up = LearnedUpsampler(hidden_dim=128, factor=3)
        x = torch.randn(2, 10, 128)
        lengths = torch.tensor([10, 7])
        out, new_lengths = up(x, lengths)
        assert out.shape == (2, 30, 128)
        assert new_lengths.tolist() == [30, 21]


class TestTextCTCTagger:
    @pytest.fixture
    def model(self):
        cfg = {
            'char_vocab_size': 100,
            'vocab_size': 500,
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'ffn_dim': 256,
            'dropout': 0.1,
            'upsample_factor': 2,
            'max_text_length': 64,
            'causal': False,
            'lr': 1e-3,
            'warmup_steps': 10,
            'max_steps': 100,
        }
        return TextCTCTagger(cfg)

    def test_forward_shape(self, model):
        char_ids = torch.randint(1, 100, (2, 15))
        char_lengths = torch.tensor([15, 12])
        log_probs, encoded_lengths = model(char_ids, char_lengths)
        assert log_probs.shape == (2, 30, 501)  # 15*2=30, vocab+blank=501
        assert encoded_lengths.tolist() == [30, 24]

    def test_greedy_decode(self, model):
        char_ids = torch.randint(1, 100, (1, 10))
        char_lengths = torch.tensor([10])
        log_probs, encoded_lengths = model(char_ids, char_lengths)
        decoded = model._greedy_decode(log_probs, encoded_lengths)
        assert len(decoded) == 1
        assert isinstance(decoded[0], list)

    def test_training_step(self, model):
        char_ids = torch.randint(1, 100, (4, 10))
        char_lengths = torch.tensor([10, 8, 10, 7])
        token_ids = torch.randint(0, 500, (4, 5))
        token_lengths = torch.tensor([5, 4, 5, 3])
        batch = (char_ids, char_lengths, token_ids, token_lengths)
        loss = model.training_step(batch, 0)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_causal_mode(self):
        cfg = {
            'char_vocab_size': 100,
            'vocab_size': 500,
            'embed_dim': 64,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
            'ffn_dim': 256,
            'dropout': 0.0,
            'upsample_factor': 2,
            'max_text_length': 64,
            'causal': True,
            'lr': 1e-3,
            'warmup_steps': 10,
            'max_steps': 100,
        }
        model = TextCTCTagger(cfg)
        char_ids = torch.randint(1, 100, (2, 10))
        char_lengths = torch.tensor([10, 8])
        log_probs, _ = model(char_ids, char_lengths)
        assert log_probs.shape[1] == 20

    def test_blank_id(self, model):
        assert model.blank_id == 500

    def test_configure_optimizers(self, model):
        opt_cfg = model.configure_optimizers()
        assert 'optimizer' in opt_cfg
        assert 'lr_scheduler' in opt_cfg
