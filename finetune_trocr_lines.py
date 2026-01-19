import os
import re
import csv
import glob
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# Finetune TrOCR model on line image + text pairs from CSV files.

# -----------------------
# text normalization
# -----------------------
def normalize_label(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)                  # collapse whitespace
    text = re.sub(r"\s+([,:;?.])", r"\1", text)       # no space before punct
    text = re.sub(r"([,:;?.])\s*", r"\1 ", text)      # exactly one space after
    return text.strip()

# -----------------------
# dataset
# -----------------------
class LineOCRDataset(Dataset):
    def __init__(self, df, processor, max_target_len=128):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        text = row["text"]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# -----------------------
# load + merge CSVs
# -----------------------
def load_all_csvs(finetune_dir="finetune"):
    paths = sorted(glob.glob(os.path.join(finetune_dir, "*_trocr.csv")))
    if not paths:
        raise RuntimeError("No *_trocr.csv files found")

    dfs = []
    for p in paths:
        df = pd.read_csv(
            p,
            engine="python",      # robust against commas in text
            on_bad_lines="warn",
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # required columns
    required = {"image_path", "line_id", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    # clean
    df["image_path"] = df["image_path"].astype(str)
    df["text"] = df["text"].astype(str).map(normalize_label)
    df = df[df["text"].str.len() > 0]
    df = df[df["image_path"].apply(os.path.exists)]

    return df

# -----------------------
# main
# -----------------------
def main():
    df = load_all_csvs("finetune")

    # stable split using line_id (important for reproducibility)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = max(1, int(0.05 * len(df)))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]

    print(f"Train lines: {len(train_df)}")
    print(f"Val lines:   {len(val_df)}")

    # model_name = "microsoft/trocr-large-handwritten"
    model_name = "microsoft/trocr-base-handwritten"
    
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 256
    model.config.num_beams = 4

    train_ds = LineOCRDataset(train_df, processor)
    val_ds = LineOCRDataset(val_df, processor)

    args = Seq2SeqTrainingArguments(
        output_dir="trocr_finetuned",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=10,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        predict_with_generate=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=processor,
    )

    trainer.train()

    trainer.save_model("trocr_finetuned/model")
    processor.save_pretrained("trocr_finetuned/processor")

if __name__ == "__main__":
    main()
