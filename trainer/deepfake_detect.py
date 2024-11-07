import lightning as L
import torch
from torchmetrics.functional.classification import accuracy
# from torcheval.metrics.functional.aggregation.auc import auc
from torchmetrics.classification import BinaryROC
import torch.nn as nn
from transformers import DetrForObjectDetection
from sklearn.metrics import auc
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


class DetectModule(L.LightningModule):
    def __init__(
            self,
            optimizer_args: dict,
            num_classes: int,
            num_queries: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        # config for the optimizer and scheduler
        self.optimizer_args = optimizer_args
        self.num_classes = num_classes
        self.input_dim = num_queries * 256
        self.mlp = MLP(self.input_dim, num_classes)

        id2label = {0: 'fake', 1: 'real'}
        label2id = {'fake': 0, 'real': 1}
        self.model = DetrForObjectDetection.from_pretrained(
            "./detr-resnet-50",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            num_queries=num_queries,
        )
        # for param in self.model.model.backbone.parameters():
        #     param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.auc = BinaryROC(thresholds=None)

    def forward(self, images, labels=None):
        # images: (batch_size, num_channel, height, width)
        # apply default mask (batch_size, height, width) where all values set to 1
        mask_shape = (images.shape[0], images.shape[2], images.shape[3])
        default_pixel_mask = torch.ones(mask_shape, device=self.device)

        model_output = self.model(pixel_values=images, pixel_mask=default_pixel_mask, labels=labels)
        # for key, value in model_output.loss_dict.items():
        #     # if torch.isnan(value).any() or torch.isinf(value).any():
        #     print(f"Key '{key}' contains a tensor with values: {value}")
        return model_output

    def training_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "train",batch_idx)

    def validation_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "val", batch_idx)

    def test_step(self, batched_inputs, batch_idx):
        return self.evaluate(batched_inputs, "test",batch_idx)

    def predict_step(self, batched_inputs, batch_idx):
        images, label = batched_inputs
        model_output = self.model(images)
        print(f"Debug model_output: {model_output}")

    # def evaluate(self, batch, stage,batch_idx):
    #     x, y = batch
    #
    #     # Convert y into the format expected by DETR.
    #     #
    #     # labels (`List[Dict]` of len `(batch_size,)`, *optional*):
    #     #   Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
    #     #   following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
    #     #   respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
    #     #   in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
    #     labels = []
    #     for img_labels, img_boxes in zip(y["labels"], y["boxes"]):
    #         labels.append({
    #             "class_labels": img_labels,
    #             "boxes": img_boxes,
    #         })
    #     detr_output = self(x, labels)  # output class: DetrObjectDetectionOutput
    #     loss = detr_output.loss
    #     #Calculate the soft logits
    #     soft_logits = nn.Softmax(dim=-1)(detr_output.logits[:, :, :2]) #(batch_size, num_queries, num_classes)
    #     # DETR output logit:
    #     # logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
    #     #             Classification logits (including no-object) for all queries.
    #
    #     # TODO: How can logit be converted to a classification logit?
    #     # [Option 1] pick the maximum logit among queries, get (batch_size, num_classes + 1)
    #     # my_pred = torch.max(detr_output.logits[:, :, :2], dim=1)[0]
    #     # get final pred among classes, (batch_size, 1)
    #     # my_pred = torch.max(my_pred, dim=1, keepdim=True)[1]
    #
    #     # [Option 2] voting by respective queries
    #     my_pred = torch.max(soft_logits, dim=2)[1]  # (batch_size, num_queries)
    #     # print(f"Debug my_pred 1: {my_pred}")
    #     # print(f"Debug my_pred 1: {my_pred.shape}")
    #     single_logit=[]
    #     if my_pred.shape[1] > 1:
    #         my_pred = torch.sum(my_pred, dim=1) / my_pred.shape[1]  # (batch_size, 1)
    #     # print(f"Debug my_pred 2: {my_pred}")
    #     # print(f"Debug vote_counts: {my_pred.shape}")
    #         my_pred = torch.where(my_pred >= 0.5, 1, 0)  # (batch_size)
    #         my_pred = my_pred.unsqueeze(1)  # (batch_size, 1)
    #         for i in range(my_pred.shape[0]):
    #             single = torch.mean(soft_logits[i][:,my_pred[i]]) # (1)
    #             single_logit.append(single)
    #         select = torch.Tensor(single_logit)
    #     else:
    #         # Pick the logit of corresponding class
    #         select = soft_logits[torch.arange(my_pred.shape[0]).unsqueeze(1), :, my_pred]
    #
    #
    #     # print(f"Debug my_pred 3: {my_pred}")
    #     # print(f"Debug my_pred 4: {my_pred.shape}")
    #
    #     # compute accuracy
    #     # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
    #     acc1 = accuracy(my_pred, y["labels"], task="multiclass", num_classes=self.num_classes, top_k=1)
    #     fpr, tpr, tresholds = self.auc(select.view(1,select.shape[0]), y["labels"].view(1,y["labels"].shape[0]))
    #     auc1 = auc(fpr.cpu().numpy(),tpr.cpu().numpy())
    #     if stage == 'train' and batch_idx % 100 == 0:
    #         fig_, ax_ = self.auc.plot(score=True)
    #         fig_.savefig(self.logger.log_dir + f"/epoch_{self.current_epoch}_step_{batch_idx}_AUC")
    #     # log every metric
    #     self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     self.log(f'{stage}_auc1', auc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     return loss

    def evaluate(self, batch, stage,batch_idx):
        x, y = batch

        # Convert y into the format expected by DETR.
        #
        # labels (`List[Dict]` of len `(batch_size,)`, *optional*):
        #   Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
        #   following 2 keys: 'class_labels' and 'boxes' (the class labels and bounding boxes of an image in the batch
        #   respectively). The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes
        #   in the image,)` and the boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 4)`.
        labels = []
        for obj_labels, img_boxes in zip(y["labels"], y["boxes"]):
            labels.append({
                "class_labels": obj_labels,
                "boxes": img_boxes,
            })
        detr_output = self(x, labels)  # output class: DetrObjectDetectionOutput
        loss1 = detr_output.loss
        last_hidden_state = detr_output.last_hidden_state.view(x.shape[0],-1)
        mlp_output = self.mlp(last_hidden_state)
        loss2 = self.criterion(mlp_output, y["class"].squeeze())
        # compute accuracy
        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
        acc1 = accuracy(mlp_output.argmax(1), y["class"].squeeze(), task="multiclass", num_classes=self.num_classes, top_k=1)

        score = torch.softmax(mlp_output.detach(), dim=-1)
        score = torch.select(score,1,1).unsqueeze(0)
        fpr, tpr, tresholds = self.auc(score, y["class"].view(1,y["class"].shape[0]))
        auc1 = auc(fpr.cpu().numpy(),tpr.cpu().numpy())
        # if stage == 'train' and batch_idx % 100 == 0:
        #     fig_, ax_ = self.auc.plot(score=True)
        #     fig_.savefig(self.logger.log_dir + f"/epoch_{self.current_epoch}_step_{batch_idx}_AUC")
        # log every metric
        self.log(f'{stage}_detr_loss', loss1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_mlp_loss', loss2, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_auc1', auc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss1 + loss2

    def configure_optimizers(self):
        # https://github-.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
        lr = float(self.optimizer_args["lr"])
        lr_backbone = float(self.optimizer_args["lr_backbone"])
        lr_weight_decay = float(self.optimizer_args["lr_weight_decay"])
        lr_mlp = float(self.optimizer_args["lr_mlp"])

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and "mlp" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
            {
                "params": [p for n, p in self.named_parameters() if "mlp" in n and p.requires_grad],
                "lr": lr_mlp,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=lr_weight_decay)
        # optimizer2 = torch.optim.Adam([{"params": self.mlp.parameters()}], lr=lr_mlp, weight_decay=lr_weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        # return [
        #     {"optimizer": optimizer1, "lr_scheduler": scheduler1},
        #     {"optimizer": optimizer2, "lr_scheduler": scheduler2},
        # ]
