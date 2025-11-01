import logging
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE


class CustomEncDecCTCModelBPE(EncDecCTCModelBPE):
    def setup_custom_loss(self):
        self.use_keyword_loss = self.cfg.get('use_keyword_loss', False)
        self.keyword_loss_weight = self.cfg.get('keyword_loss_weight', 0.3)
        self.keyword_loss_warmup_steps = self.cfg.get('keyword_loss_warmup_steps', 0)
        self.keyword_token_ids = set()

    def set_keyword_token_ids(self, keyword_ids):
        self.keyword_token_ids = set(keyword_ids)
        logging.info(f"Set {len(self.keyword_token_ids)} keyword token IDs for custom loss.")

    def training_step(self, batch, batch_idx):
        audio_signal, audio_signal_len, transcript, transcript_len = batch
        log_probs, encoded_len, greedy_predictions = self.forward(
            input_signal=audio_signal, input_signal_length=audio_signal_len
        )

        loss_value = self.loss(
            log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
        )

        if self.use_keyword_loss and self.training and len(self.keyword_token_ids) > 0:
            current_step = self.trainer.global_step
            if self.keyword_loss_warmup_steps > 0 and current_step < self.keyword_loss_warmup_steps:
                current_keyword_loss_weight = (
                    current_step / self.keyword_loss_warmup_steps
                ) * self.keyword_loss_weight
            else:
                current_keyword_loss_weight = self.keyword_loss_weight

            self.log('current_keyword_loss_weight', current_keyword_loss_weight, on_step=True, on_epoch=False, prog_bar=True)

            keyword_targets = []
            keyword_target_lengths = []
            for i in range(transcript.size(0)):
                target = transcript[i][: transcript_len[i]]
                keyword_target = [
                    token_id.item() for token_id in target if token_id.item() in self.keyword_token_ids
                ]

                if len(keyword_target) > 0:
                    keyword_targets.append(
                        torch.tensor(keyword_target, dtype=torch.long, device=transcript.device)
                    )
                    keyword_target_lengths.append(len(keyword_target))

            if keyword_targets:
                keyword_targets_padded = torch.nn.utils.rnn.pad_sequence(
                    keyword_targets, batch_first=True, padding_value=self.tokenizer.pad_id
                )
                keyword_target_lengths_tensor = torch.tensor(
                    keyword_target_lengths, dtype=torch.long, device=transcript.device
                )

                if keyword_targets_padded.numel() > 0:
                    keyword_loss = self.loss(
                        log_probs=log_probs,
                        targets=keyword_targets_padded,
                        input_lengths=encoded_len,
                        target_lengths=keyword_target_lengths_tensor,
                    )
                    loss_value = (
                        1 - current_keyword_loss_weight
                    ) * loss_value + current_keyword_loss_weight * keyword_loss

        self.log('train_loss', loss_value)
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        return {'loss': loss_value}
