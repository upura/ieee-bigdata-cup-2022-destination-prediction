import dataclasses
import sys
from collections import OrderedDict
from typing import List

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


def preprocessing_df(input_df):
    output_df = input_df.copy()
    # convert to zero-origin
    output_df["Gender"] = output_df["Gender"] - 1
    output_df["Age"] = output_df["Age"] - 1
    output_df["Occupation"] = output_df["Occupation"] - 1
    output_df["Trip_type"] = output_df["Trip_type"] - 1
    # extract datetime features
    # todo: holiday info
    output_df["Departure_time"] = pd.to_datetime(output_df["Departure_time"])
    output_df["Departure_hour"] = output_df["Departure_time"].dt.hour
    output_df["Departure_dow"] = output_df["Departure_time"].dt.weekday
    return output_df


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = MyModel(
            numerical_cols=cfg.numerical_cols,
            categorical_cols=cfg.categorical_cols,
            cat_dims=cfg.cat_dims,
            num_classes=cfg.NUM_CLASSES,
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        output = self.backbone(x["x_num"], x["x_cat"])
        return output

    def training_step(self, batch, batch_idx):
        x_num = batch["x_num"]
        x_cat = batch["x_cat"]
        targets = batch["y"]
        output = self.backbone(x_num, x_cat)
        loss = self.criterion(output, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x_num = batch["x_num"]
        x_cat = batch["x_cat"]
        targets = batch["y"]
        output = self.backbone(x_num, x_cat)
        loss = self.criterion(output, targets)
        output = OrderedDict(
            {
                "targets": targets.detach(),
                "preds": output.detach(),
                "loss": loss.detach(),
            }
        )
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"] for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"] for o in outputs]).cpu().numpy()
        score = accuracy_score(np.argmax(targets, 1), np.argmax(preds, 1))
        d["v_score"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


class TabularDataset(Dataset):
    def __init__(
        self,
        df,
        numerical_cols: List[str],
        categorical_cols: List[str],
        target_col: str,
        num_classes: int,
        is_train: bool = True,
    ):
        super().__init__()

        self.df = df
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col
        self.is_train = is_train
        self.num_classes = num_classes

        if self.is_train:
            self.X = self.df.drop(self.target_col, axis=1).copy()
            self.y = self.df[self.target_col].values
        else:
            self.X = self.df.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.is_train:

            labels = self.y[index]
            targets = np.zeros(self.num_classes, dtype=float)
            targets[labels] = 1.0
            x_num = self.X.loc[index, self.numerical_cols].values.astype(np.float32)
            x_cat = self.X.loc[index, self.categorical_cols].values.astype(np.int32)

            return {
                "x_num": torch.FloatTensor(x_num),
                "x_cat": torch.LongTensor(x_cat),
                "y": torch.FloatTensor(targets),
                "labels": torch.tensor(labels),
            }

        else:

            x_num = self.X.loc[index, self.numerical_cols].values.astype(np.float32)
            x_cat = self.X.loc[index, self.categorical_cols].values.astype(np.int32)

            return {
                "x_num": torch.FloatTensor(x_num),
                "x_cat": torch.LongTensor(x_cat),
            }


class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.test_df = None
        self.train_df = None
        self.valid_df = None
        self.cfg = cfg

    def get_test_df(self):
        df = pd.read_csv(self.cfg.TEST_DF_PATH)
        return df

    def get_dev_df(self):
        df = pd.read_csv(self.cfg.DEV_DF_PATH)
        if int(self.cfg.debug):
            return df.sample(1000).reset_index(drop=True)
        else:
            return df

    def get_train_df(self):
        df = pd.read_csv(self.cfg.TRAIN_DF_PATH)
        if int(self.cfg.debug):
            return df.sample(1000).reset_index(drop=True)
        else:
            return df

    def setup(self, stage):
        self.train_df = self.get_train_df()
        self.valid_df = self.get_dev_df()
        self.test_df = self.get_test_df()

    def get_dataframe(self, phase):
        assert phase in {"train", "valid", "test"}
        if phase == "train":
            return self.train_df
        elif phase == "valid":
            return self.valid_df
        elif phase == "test":
            return self.test_df

    def get_ds(self, phase):
        assert phase in {"train", "valid", "test"}
        ds = TabularDataset(
            df=self.get_dataframe(phase=phase),
            numerical_cols=self.cfg.numerical_cols,
            categorical_cols=self.cfg.categorical_cols,
            target_col=self.cfg.TARGET_COL,
            num_classes=self.cfg.NUM_CLASSES,
            is_train=(phase != "test"),
        )
        return ds

    def get_loader(self, phase):
        dataset = self.get_ds(phase=phase)
        return DataLoader(
            dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=(phase == "train"),
            num_workers=self.cfg.NUM_WORKERS,
            drop_last=(phase == "train"),
        )

    # Trainer.fit() 時に呼び出される
    def train_dataloader(self):
        return self.get_loader(phase="train")

    # Trainer.fit() 時に呼び出される
    def val_dataloader(self):
        return self.get_loader(phase="valid")

    def test_dataloader(self):
        return self.get_loader(phase="test")


class MyModel(nn.Module):
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        cat_dims: List[int],
        num_classes: List[int],
    ):
        super().__init__()
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.num_classes = num_classes
        self.hidden_size = 100
        self.emb_dim = 50
        self.cat_dims = cat_dims
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.cat_dims[idx], self.emb_dim)
                for idx in range(len(self.categorical_cols))
            ]
        )
        self.cate_proj = nn.Sequential(
            nn.Linear(self.emb_dim * len(self.categorical_cols), self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
        )
        self.cont_emb = nn.Sequential(
            nn.Linear(len(self.numerical_cols), self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, x_num, x_cat):
        num_o = self.cont_emb(x_num)
        cat_o = self.cate_proj(
            torch.cat(
                [self.cat_embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])],
                dim=-1,
            )
        )
        outputs = self.ffn(torch.cat([num_o, cat_o], axis=1))
        return outputs


@dataclasses.dataclass
class Cfg:
    PROJECT_NAME = "bigdata"
    RUN_NAME = "exp000"
    NUM_CLASSES = 2
    NUM_EPOCHS = 3
    NUM_WORKERS = 4
    BATCH_SIZE = 512
    OUTPUT_PATH = "."
    TRAIN_DF_PATH = "train.csv"
    TEST_DF_PATH = "../input/ieee-bigdata-prepare-dataset/test.csv"
    TARGET_COL = "target"


Cfg.numerical_cols = [
    "T000918002_ori",
    "T000918006_ori",
    "T000918021_ori",
    "T000918025_ori",
    "T000847001_ori",
    "T000918002_des",
    "T000918006_des",
    "T000918021_des",
    "T000918025_des",
    "T000847001_des",
]
Cfg.categorical_cols = [
    "Gender",
    "Age",
    "Occupation",
    "Trip_type",
    "Departure_hour",
    "Departure_dow",
]

if __name__ == "__main__":

    if "google.colab" in sys.modules:
        secret_value = "YOUR_SECRET"
    else:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=secret_value)

    fold = 0
    debug = False
    cfg = Cfg()
    cfg.fold = fold
    cfg.debug = debug

    train_tokyo = pd.read_csv("../input/ieee-bigdata-prepare-dataset/train_tokyo.csv")
    train_chukyo = pd.read_csv("../input/ieee-bigdata-prepare-dataset/train_chukyo.csv")
    train_kyushu = pd.read_csv("../input/ieee-bigdata-prepare-dataset/train_kyushu.csv")
    train_higashisurugawan = pd.read_csv(
        "../input/ieee-bigdata-prepare-dataset/train_higashisurugawan.csv"
    )

    if fold == 0:
        train = pd.concat([train_chukyo, train_kyushu, train_higashisurugawan])
        cfg.DEV_DF_PATH = "../input/ieee-bigdata-prepare-dataset/train_tokyo.csv"
    elif fold == 1:
        pass
    elif fold == 2:
        pass
    elif fold == 3:
        pass
    train.reset_index(drop=True).to_csv("train.csv", index=False)

    # dev = pd.read_csv(cfg.DEV_DF_PATH)
    # test = pd.read_csv(cfg.TEST_DF_PATH)
    # train_test = pd.concat([train, dev, test], sort=False)
    # cfg.cat_dims = [int(train_test[col].nunique()) + 1 for col in cfg.categorical_cols]
    cfg.cat_dims = [3, 18, 21, 24, 25, 7]
    # print(cfg.cat_dims)

    seed_everything(777)
    logger = CSVLogger(save_dir=str(cfg.OUTPUT_PATH), name=f"fold_{fold}")
    wandb_logger = WandbLogger(name=f"{cfg.RUN_NAME}_{fold}", project=cfg.PROJECT_NAME)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.OUTPUT_PATH),
        filename=f"{cfg.RUN_NAME}_fold_{fold}",
        save_weights_only=True,
        monitor=None,
    )
    trainer = Trainer(
        max_epochs=cfg.NUM_EPOCHS,
        gpus=(-1 if torch.cuda.is_available() else None),
        callbacks=[checkpoint_callback],
        logger=[logger, wandb_logger],
    )

    model = MyLightningModule(cfg)
    datamodule = MyDataModule(cfg)
    trainer.fit(model, datamodule=datamodule)

    y_val_pred = torch.cat(trainer.predict(model, datamodule.val_dataloader()))
    np.save(f"y_val_pred_fold{fold}", y_val_pred.to("cpu").detach().numpy())

    y_test_pred = torch.cat(trainer.predict(model, datamodule.test_dataloader()))
    np.save(f"y_test_pred_fold{fold}", y_test_pred.to("cpu").detach().numpy())
