import lightning as L
import torch
from torchmetrics.functional.classification import accuracy
# from torcheval.metrics.functional.aggregation.auc import auc
from torchmetrics.classification import BinaryROC
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import knn_graph
from torch_geometric.nn import GCNConv, global_mean_pool,GraphNorm
from utility import minimize_iou_overlap_loss, size_constraint_loss,convert_boxes_format,extract_and_create_graph_per_sample, extract_and_normalize_features,compute_cosine_similarity,similarity_loss
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


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNClassifier, self).__init__()
        # 定义 GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 定义 GraphNorm
        self.graph_norm1 = GraphNorm(hidden_dim)
        self.graph_norm2 = GraphNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x, batch).relu()
        # x = self.conv2(x, edge_index)
        # x = self.graph_norm2(x, batch).relu()
        # 使用 global_mean_pool，根据 batch 进行池化
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DetectModule(L.LightningModule):
    def __init__(
            self,
            optimizer_args: dict,
            num_classes: int,
            num_queries: int,
            cropped_size: int,
            freeze_encoder: bool,
            freeze_decoder: bool,
            overlap_loss_weight: int,
            pretrained: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        # config for the optimizer and scheduler
        self.optimizer_args = optimizer_args
        self.num_classes = num_classes
        self.input_dim = num_queries * 256
        self.mlp = MLP(self.input_dim, num_classes)
        self.k = cropped_size

        self.overlap_loss_weight = overlap_loss_weight


        if pretrained:
            # id2label = {0: 'fake', 1: 'real'}
            # label2id = {'fake': 0, 'real': 1}
            # id2label = {0: 'f_area1', 1: 'f_area2', 2: 'f_area3', 3: 'f_area4', 4: 'r_area1', 5: 'r_area2', 6: 'r_area3', 7: 'r_area4'}
            # label2id = {'f_area1': 0, 'f_area2': 1,  'f_area3': 2, 'f_area4': 3, 'r_area1': 4, 'r_area2': 5, 'r_area3': 6, 'r_area4': 7}
            id2label = {0: 'df_1', 1: 'df_2', 2: 'df_3', 3: 'df_4', 4: 'f2f_1', 5: 'f2f_2', 6: 'f2f_3', 7: 'f2f_4',
                        8: 'fs_1', 9: 'fs_2', 10: 'fs_3', 11: 'fs_4', 12: 'nt_1', 13: 'nt_2', 14: 'nt_3', 15:
                        'nt_4', 16: 'real_1', 17: 'real_2', 18: 'real_3', 19: 'real_4'}
            label2id = {'df_1': 0, 'df_2': 1, 'df_3': 2, 'df_4': 3, 'f2f_1': 4, 'f2f_2': 5, 'f2f_3': 6, 'f2f_4': 7,
                        'fs_1': 8, 'fs_2': 9, 'fs_3': 10, 'fs_4': 11, 'nt_1': 12, 'nt_2': 13, 'nt_3': 14, 'nt_4': 15,
                        'real_1': 16, 'real_2': 17, 'real_3': 18, 'real_4': 19}

            self.model = DetrForObjectDetection.from_pretrained(
                "./detr-resnet-50",
                id2label= id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                num_queries=num_queries,
            )
            self.config = self.model.config
            # for param in self.model.model.backbone.parameters():
            #     param.requires_grad = False
            # Freeze the parameters of the encoder
            if freeze_encoder:
                for param in self.model.model.encoder.parameters():
                    param.requires_grad = False
            else:
                print("Encoder not freezed")

            if freeze_decoder:
                # Freeze the parameters of the decoder
                for param in self.model.model.decoder.parameters():
                    param.requires_grad = False
            else:
                print("Decoder not freezed")
        else:
            self.config = DetrConfig.from_json_file("/home/bosheng/deepfake-detect-main/config.json")
            self.model = DetrForObjectDetection(self.config)
            print("Training the DETR from scratch!")




        self.feature_maps = {}
        # 存储输出形状
        # self.feature_shape = None
        self.target_layer = self.model.model.backbone.conv_encoder.model.layer1[1].act1 #layer4[-1].conv3  layer1[0].conv3 conv1
        self.hook_handle = self.target_layer.register_forward_hook(self.hook_fn)
        self.gnn = GNNClassifier(int(self.k * self.k*self.model.model.backbone.conv_encoder.model.layer1[1].conv1.weight.shape[0]), 128, num_classes)
        # self.conv1x1 = nn.Conv2d(self.target_layer.weight.shape[0], 1, kernel_size=1)
        self.criterion = nn.CrossEntropyLoss()
        self.auc = BinaryROC(thresholds=None)

    def hook_fn(self, module, input, output):
        # feature_maps = {}
        # self.feature_shape = output.shape
        self.feature_maps['feats'] = output.detach()
    def forward(self, images, labels=None):
        # images: (batch_size, num_channel, height, width)
        # apply default mask (batch_size, height, width) where all values set to 1
        mask_shape = (images.shape[0], images.shape[2], images.shape[3])
        default_pixel_mask = torch.ones(mask_shape, device=self.device)

        model_output = self.model(pixel_values=images, pixel_mask=default_pixel_mask, labels=labels)

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
    #Major voting (version1)
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

    #MLP(version2)
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
    #     for obj_labels, img_boxes in zip(y["labels"], y["boxes"]):
    #         labels.append({
    #             "class_labels": obj_labels,
    #             "boxes": img_boxes,
    #         })
    #     detr_output = self(x, labels)  # output class: DetrObjectDetectionOutput
    #     loss1 = detr_output.loss
    #     last_hidden_state = detr_output.last_hidden_state.view(x.shape[0],-1)
    #     mlp_output = self.mlp(last_hidden_state)
    #     loss2 = self.criterion(mlp_output, y["class"].squeeze())
    #     # compute accuracy
    #     # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
    #     acc1 = accuracy(mlp_output.argmax(1), y["class"].squeeze(), task="multiclass", num_classes=self.num_classes, top_k=1)
    #
    #     score = torch.softmax(mlp_output.detach(), dim=-1)
    #     score = torch.select(score,1,1).unsqueeze(0)
    #     fpr, tpr, tresholds = self.auc(score, y["class"].view(1,y["class"].shape[0]))
    #     auc1 = auc(fpr.cpu().numpy(),tpr.cpu().numpy())
    #     # if stage == 'train' and batch_idx % 100 == 0:
    #     #     fig_, ax_ = self.auc.plot(score=True)
    #     #     fig_.savefig(self.logger.log_dir + f"/epoch_{self.current_epoch}_step_{batch_idx}_AUC")
    #     # log every metric
    #     self.log(f'{stage}_detr_loss', loss1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     self.log(f'{stage}_mlp_loss', loss2, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     self.log(f'{stage}_auc1', auc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    #     return loss1 + loss2
    # KNN with GNN (version3)
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
        box_size_loss = size_constraint_loss(detr_output['pred_boxes'])
        iou_overlap_loss = self.overlap_loss_weight * minimize_iou_overlap_loss(detr_output['pred_boxes'], max_iou=0.3)

        #提取hook的特征图
        extracted_features = self.feature_maps['feats']

        # 提取并归一化特征
        # normalized_features = extract_and_normalize_features(extracted_features, y["boxes"],x.shape[2:],self.device)
        # similarity_matrices = compute_cosine_similarity(normalized_features)
        #
        # # 2. 计算相似度惩罚损失
        # sim_loss = 4 * similarity_loss(similarity_matrices)

        # extracted_features_1ch = extracted_features.mean(dim=1, keepdim=True) #降维_直接取均值
        # extracted_features_1ch = self.conv1x1(extracted_features)#降维_通过卷积
        # 获取预测的边界框
        pred_boxes = detr_output['pred_boxes']
        converted_boxes = convert_boxes_format(y['boxes'])
        all_boxes = torch.cat([converted_boxes,pred_boxes],dim=1)
        # # 调整尺寸
        output_size = (self.k,self.k)  # 例如 (7, 7)

        data = extract_and_create_graph_per_sample(extracted_features,converted_boxes,self.device, k=5, output_size=output_size)


        data.to(self.device)

        gnn_output = self.gnn(data)

        loss2 = self.criterion(gnn_output, y["class"].squeeze())
        # compute accuracy
        # https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html
        acc1 = accuracy(gnn_output.argmax(1), y["class"].squeeze(), task="multiclass", num_classes=self.num_classes, top_k=1)

        score = torch.softmax(gnn_output.detach(), dim=-1)
        score = torch.select(score,1,1).unsqueeze(0)
        fpr, tpr, tresholds = self.auc(score, y["class"].view(1,y["class"].shape[0]))
        auc1 = auc(fpr.cpu().numpy(),tpr.cpu().numpy())
        # log every metric
        self.log(f'{stage}_detr_loss', loss1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_detr_ce_loss', detr_output['loss_dict']['loss_ce'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_detr_box_loss', detr_output['loss_dict']['loss_bbox'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_detr_giou_loss', detr_output['loss_dict']['loss_giou'], on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_detr_iou_overlap_loss', iou_overlap_loss, on_step=True, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log(f'{stage}_detr_box_size_loss', box_size_loss, on_step=True, on_epoch=True,
                 logger=True, sync_dist=True)
        # self.log(f'{stage}_detr_box_innerSim_loss', sim_loss, on_step=True, on_epoch=True,
        #          logger=True, sync_dist=True)
        self.log(f'{stage}_gnn_loss', loss2, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_acc1', acc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log(f'{stage}_auc1', auc1, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss1 + self.config.gnn_loss_coefficient * loss2 + iou_overlap_loss + box_size_loss #+ sim_loss

    def configure_optimizers(self):
        # https://github-.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
        lr = float(self.optimizer_args["lr"])
        lr_backbone = float(self.optimizer_args["lr_backbone"])
        lr_weight_decay = float(self.optimizer_args["lr_weight_decay"])
        lr_clr = float(self.optimizer_args["lr_clr"])

        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and "gnn" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
            {
                "params": [p for n, p in self.named_parameters() if "gnn" in n and p.requires_grad],
                "lr": lr_clr,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=lr, weight_decay=lr_weight_decay)
        # optimizer = torch.optim.SGD(param_dicts, lr=lr, momentum=0.9)
        # optimizer = torch.optim.Adam(param_dicts, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        # schedulercccc2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

        #改源代码的网址： https://github.com/facebookresearch/detr/issues/101