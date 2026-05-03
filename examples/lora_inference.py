import torch
from peft import PeftModel

from CRISCross.models import CRISCross


def load_lora_model(base_url: str, lora_weights_path: str, cfg: dict) -> PeftModel:
    """Load a CRISCross model with a saved LoRA adapter.

    Args:
        base_url: URL to the base pretrained state_dict (same URL used during training).
        lora_weights_path: Local directory containing the saved LoRA adapter
                           (produced by peft_model.save_pretrained()).
        cfg: Dict of CRISCross constructor kwargs matching the training configuration.

    Returns:
        PeftModel in eval mode with LoRA weights loaded on top of the base model.
    """
    base_model = CRISCross(**cfg)
    state_dict = torch.hub.load_state_dict_from_url(base_url)
    base_model.load_state_dict(state_dict)

    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()
    return model


if __name__ == "__main__":
    PRETRAINED_URL = "https://huggingface.co/domonik/criscross-atac/resolve/main/model.pt"
    # Path to the directory saved by lora_finetuning.py
    LORA_WEIGHTS_PATH = "lora_adapter"

    cfg = {
        "vocab_size": 5,
        "dropout": 0.2,
        "context_layers": 3,
        "hidden_dim": 512,
        "num_epi": 1,
        "output_size": 1,
        "windowsize": 512,
        "merge": "early",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_lora_model(PRETRAINED_URL, LORA_WEIGHTS_PATH, cfg)
    model = model.to(device)

    print("LoRA model loaded successfully")
    model.print_trainable_parameters()

    # Example inference with synthetic data.
    # In practice, replace these tensors with your GenomicDataModule batches.
    batch_size = 2
    windowsize = 512
    num_epi = 1

    target = torch.randint(0, 5, (batch_size, 25), device=device)
    off_target = torch.randint(0, 5, (batch_size, windowsize), device=device)
    strand = torch.randint(0, 2, (batch_size,), device=device)
    # epi shape: [batch, windowsize, num_epi] — last dim is num_epi for the Linear layer
    epi = torch.randn(batch_size, windowsize, num_epi, device=device)

    with torch.no_grad():
        logits, _ = model(target, off_target, strand, epi)
        probs = torch.sigmoid(logits)

    print(f"Logits : {logits.squeeze(-1).tolist()}")
    print(f"Probs  : {probs.squeeze(-1).tolist()}")

