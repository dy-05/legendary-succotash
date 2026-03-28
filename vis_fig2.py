import torch
import torch.nn as nn
import sys

sys.path.append("..")

import clip
from prompts.imagenet_template import openai_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from pamr import PAMR

from torchvision import transforms
import torch.nn.functional as F

from segment_anything import sam_model_registry
import dino.vision_transformer as vits
from open_clip import create_model_and_transforms, tokenizer
import numpy as np


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


class CLIPForSegmentation(BaseSegmentor):
    def __init__(self, clip_path, name_path, device=torch.device('cuda'),
                 pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40,
                 slide_stride=112, slide_crop=336):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True,
        )
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _, preprocess = create_model_and_transforms('ViT-B/16', pretrained='openai', precision='fp32')
        self.net.to(device).eval()

        # self.sam = sam_model_registry["vit_l"](checkpoint='../sam_vit_l_0b3195.pth')
        self.sam = sam_model_registry["vit_b"](checkpoint='../sam_vit_b_01ec64.pth')
        for p in self.sam.parameters():
            p.requires_grad = False
        self.sam = self.sam.to(device).eval()

        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        for p in self.dinov2.parameters():
            p.requires_grad = False
        self.dinov2.to(device).eval()

        # self.dino = vits.__dict__["vit_small"](patch_size=16, num_classes=0)
        # state_dict = torch.load('../dino_deitsmall16_pretrain.pth')

        self.dino = vits.__dict__["vit_base"](patch_size=8, num_classes=0)
        state_dict = torch.load('../dino_vitbase8_pretrain.pth')

        # self.dino = vits.__dict__["vit_base"](patch_size=16, num_classes=0)
        # state_dict = torch.load('../dino_vitbase16_pretrain.pth')

        self.dino.load_state_dict(state_dict, strict=True)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.to(device).eval()


        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = clip.tokenize([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]

        patch_size = self.net.visual.patch_size
        w, h = img[0].shape[-2] // patch_size[0], img[0].shape[-1] // patch_size[1]

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        ## SAM
        imgs_norm_sam = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
        sam_feats, _ = self.sam.image_encoder(imgs_norm_sam)

        sam_feats = F.interpolate(sam_feats, size=(w, h), mode='bilinear', align_corners=False)
        sam_feats = sam_feats.reshape(sam_feats.shape[0], sam_feats.shape[1], -1)
        sam_feats = F.normalize(sam_feats, dim=1)
        similarity = torch.einsum("b c m, b c n -> b m n", sam_feats, sam_feats)

        # similarity = (similarity - torch.mean(similarity) * 1.3) * 3.0
        # similarity[similarity < 0.0] = float('-inf')

        similarity[similarity < 0.75] = float('-inf')  # 0.75 sam  0.5 dinov2  0.2 dino

        attn_weights_sam = F.softmax(similarity, dim=-1)

        ## DINO

        feat, attn, qkv = self.dino.get_intermediate_feat(imgs_norm)

        feat_h, feat_w = img.shape[2] // 8, img.shape[3] // 8
        dino_feats = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        dino_feats = F.interpolate(dino_feats, size=(w, h), mode='bilinear', align_corners=False)
        dino_feats = dino_feats.reshape(dino_feats.shape[0], dino_feats.shape[1], -1)
        dino_feats = F.normalize(dino_feats, dim=1)
        similarity = torch.einsum("b c m, b c n -> b m n", dino_feats, dino_feats)

        # similarity = (similarity - torch.mean(similarity) * 1.3) * 3.0
        # similarity[similarity < 0.0] = float('-inf')

        similarity[similarity < 0.5] = float('-inf')  # 0.75 sam  0.5 dinov2  0.2 dino

        attn_weights_dino = F.softmax(similarity, dim=-1)

        # DINOV2
        dinov2_feats = self.dinov2.get_intermediate_layers(imgs_norm, reshape=True)[0] # b c w h

        dinov2_feats = F.interpolate(dinov2_feats, size=(w, h), mode='bilinear', align_corners=False)
        dinov2_feats = dinov2_feats.reshape(dinov2_feats.shape[0], dinov2_feats.shape[1], -1)
        dinov2_feats = F.normalize(dinov2_feats, dim=1)
        similarity = torch.einsum("b c m, b c n -> b m n", dinov2_feats, dinov2_feats)

        # similarity = (similarity - torch.mean(similarity) * 1.3) * 3.0
        # similarity[similarity < 0.0] = float('-inf')

        similarity[similarity < 0.5] = float('-inf')  # 0.75 sam  0.5 dinov2  0.2 dino

        attn_weights_dinov2 = F.softmax(similarity, dim=-1)

        image_features, qk_attn, qq_attn, kk_attn, vv_attn = self.net.encode_image(img,
                                                                                   attn_type='qk',
                                                                                   external_feats=None,
                                                                                   attn=None,
                                                                                   beta=1.2,
                                                                                   gamma=3.0,
                                                                                   token_size=None)

        attn_weights_csa = qk_attn, qq_attn, kk_attn, vv_attn


        return attn_weights_sam, attn_weights_dino, attn_weights_dinov2, attn_weights_csa

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            with torch.no_grad():
                seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        # return self.postprocess_result(seg_logits, data_samples)
        return seg_logits

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            data_samples[i].set_data({
                'seg_logits':
                    PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                    PixelData(**{'data': seg_pred})
            })

        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices



if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    import cv2
    img = Image.open('../images/horses.jpg')
    normalize1 = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),

    ])

    normalize2 = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

    img = normalize1(img).unsqueeze(0).to('cuda')

    model = CLIPForSegmentation(clip_path='ViT-B/16', name_path='./configs/cls_voc21.txt', slide_crop=0)
    attn_weights_sam, attn_weights_dino, attn_weights_dinov2, attn_weights_csa = model.predict(img, None)

    qk_attn, q_attn, k_attn, v_attn = attn_weights_csa
    q_attn = torch.mean(q_attn[0:1], dim=0, keepdim=True)[:, 1:, 1:]
    k_attn = torch.mean(k_attn[0:1], dim=0, keepdim=True)[:, 1:, 1:]
    v_attn = torch.mean(v_attn[0:1], dim=0, keepdim=True)[:, 1:, 1:]
    qk_attn = torch.mean(qk_attn[0:1], dim=0, keepdim=True)[:, 1:, 1:]

    imgg = unnorm(img[0].cpu()).permute(1,2,0)
    plt.figure(figsize=(20,20))
    plt.imshow(imgg)
    plt.axis('off')
    plt.scatter(256, 280, s=1000, c='red', marker='o')   # 491
    plt.show()

    fig, ax = plt.subplots(2, 4, figsize=(4 * 10, 2 * 10))

    ax[0, 0].axis('off')
    ax[0, 0].imshow(imgg)
    ax[0, 0].scatter(256, 280, s=1500, c='red', marker='o')   # 491

    plt.figure(figsize=(20, 20))
    plt.imshow(imgg)
    plt.scatter(256, 280, s=1500, c='red', marker='o')   # 491
    plt.axis('off')
    plt.savefig('subfigures/img_1.png', bbox_inches='tight', pad_inches=0)

    ax[0, 1].axis('off')
    wh = int(np.sqrt(attn_weights_sam.size(2)))
    attn_sam = attn_weights_sam[0][491].cpu().reshape(wh, wh)
    attn_sam = F.interpolate(attn_sam.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_sam = attn_sam / np.max(attn_sam)
    heatmap = cv2.applyColorMap((attn_sam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 1].imshow(blended)
    ax[0, 1].set_title("SAM", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/sam_1.png', bbox_inches='tight', pad_inches=0)


    ax[0, 2].axis('off')
    wh = int(np.sqrt(attn_weights_dino.size(2)))
    attn_dino = attn_weights_dino[0][491].cpu().reshape(wh, wh)
    attn_dino = F.interpolate(attn_dino.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dino = attn_dino / np.max(attn_dino)
    heatmap = cv2.applyColorMap((attn_dino * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 2].imshow(blended)
    ax[0, 2].set_title("DINO", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dino_1.png', bbox_inches='tight', pad_inches=0)

    ax[0, 3].axis('off')
    wh = int(np.sqrt(attn_weights_dinov2.size(2)))
    attn_dinov2 = attn_weights_dinov2[0][491].cpu().reshape(wh, wh)
    attn_dinov2 = F.interpolate(attn_dinov2.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dinov2 = attn_dinov2 / np.max(attn_dinov2)
    heatmap = cv2.applyColorMap((attn_dinov2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 3].imshow(blended)
    ax[0, 3].set_title("DINOV2", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dinov2_1.png', bbox_inches='tight', pad_inches=0)

    ax[1, 0].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    q_attnn = q_attn[0][491].cpu().reshape(wh, wh)
    q_attnn = F.interpolate(q_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    q_attnn = q_attnn / np.max(q_attnn)
    heatmap = cv2.applyColorMap((q_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 0].imshow(blended)
    ax[1, 0].set_title("Q_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qq_1.png', bbox_inches='tight', pad_inches=0)

    ax[1, 1].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    k_attnn = k_attn[0][491].cpu().reshape(wh, wh)
    k_attnn = F.interpolate(k_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    k_attnn = k_attnn / np.max(k_attnn)
    heatmap = cv2.applyColorMap((k_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 1].imshow(blended)
    ax[1, 1].set_title("K_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/kk_1.png', bbox_inches='tight', pad_inches=0)

    ax[1, 2].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    v_attnn = v_attn[0][491].cpu().reshape(wh, wh)
    v_attnn = F.interpolate(v_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    v_attnn = v_attnn / np.max(v_attnn)
    heatmap = cv2.applyColorMap((v_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 2].imshow(blended)
    ax[1, 2].set_title("V_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/vv_1.png', bbox_inches='tight', pad_inches=0)

    ax[1, 3].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    qk_attnn = qk_attn[0][491].cpu().reshape(wh, wh).detach()
    qk_attnn = F.interpolate(qk_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear', align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    qk_attnn = qk_attnn / np.max(qk_attnn)
    heatmap = cv2.applyColorMap((qk_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 3].imshow(blended)
    ax[1, 3].set_title("QK_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qk_1.png', bbox_inches='tight', pad_inches=0)

    plt.show()




    ##########################################

    fig, ax = plt.subplots(2, 4, figsize=(4 * 10, 2 * 10))

    ax[0, 0].axis('off')
    ax[0, 0].imshow(imgg)
    ax[0, 0].scatter(180, 390, s=1500, c='red', marker='o')  # 683

    plt.figure(figsize=(20, 20))
    plt.imshow(imgg)
    plt.scatter(180, 390, s=1500, c='red', marker='o')  # 683
    plt.axis('off')
    plt.savefig('subfigures/img_2.png', bbox_inches='tight', pad_inches=0)

    ax[0, 1].axis('off')
    wh = int(np.sqrt(attn_weights_sam.size(2)))
    attn_sam = attn_weights_sam[0][683].cpu().reshape(wh, wh)
    attn_sam = F.interpolate(attn_sam.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_sam = attn_sam / np.max(attn_sam)
    heatmap = cv2.applyColorMap((attn_sam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 1].imshow(blended)
    ax[0, 1].set_title("SAM", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/sam_2.png', bbox_inches='tight', pad_inches=0)

    ax[0, 2].axis('off')
    wh = int(np.sqrt(attn_weights_dino.size(2)))
    attn_dino = attn_weights_dino[0][683].cpu().reshape(wh, wh)
    attn_dino = F.interpolate(attn_dino.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                              align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dino = attn_dino / np.max(attn_dino)
    heatmap = cv2.applyColorMap((attn_dino * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 2].imshow(blended)
    ax[0, 2].set_title("DINO", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dino_2.png', bbox_inches='tight', pad_inches=0)

    ax[0, 3].axis('off')
    wh = int(np.sqrt(attn_weights_dinov2.size(2)))
    attn_dinov2 = attn_weights_dinov2[0][683].cpu().reshape(wh, wh)
    attn_dinov2 = F.interpolate(attn_dinov2.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dinov2 = attn_dinov2 / np.max(attn_dinov2)
    heatmap = cv2.applyColorMap((attn_dinov2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 3].imshow(blended)
    ax[0, 3].set_title("DINOV2", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dinov2_2.png', bbox_inches='tight', pad_inches=0)

    ax[1, 0].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    q_attnn = q_attn[0][683].cpu().reshape(wh, wh)
    q_attnn = F.interpolate(q_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    q_attnn = q_attnn / np.max(q_attnn)
    heatmap = cv2.applyColorMap((q_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 0].imshow(blended)
    ax[1, 0].set_title("Q_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qq_2.png', bbox_inches='tight', pad_inches=0)

    ax[1, 1].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    k_attnn = k_attn[0][683].cpu().reshape(wh, wh)
    k_attnn = F.interpolate(k_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    k_attnn = k_attnn / np.max(k_attnn)
    heatmap = cv2.applyColorMap((k_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 1].imshow(blended)
    ax[1, 1].set_title("K_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/kk_2.png', bbox_inches='tight', pad_inches=0)

    ax[1, 2].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    v_attnn = v_attn[0][683].cpu().reshape(wh, wh)
    v_attnn = F.interpolate(v_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    v_attnn = v_attnn / np.max(v_attnn)
    heatmap = cv2.applyColorMap((v_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 2].imshow(blended)
    ax[1, 2].set_title("V_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/vv_2.png', bbox_inches='tight', pad_inches=0)

    ax[1, 3].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    qk_attnn = qk_attn[0][683].cpu().reshape(wh, wh).detach()
    qk_attnn = F.interpolate(qk_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    qk_attnn = qk_attnn / np.max(qk_attnn)
    heatmap = cv2.applyColorMap((qk_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 3].imshow(blended)
    ax[1, 3].set_title("QK_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qk_2.png', bbox_inches='tight', pad_inches=0)

    plt.show()

    ##################################

    fig, ax = plt.subplots(2, 4, figsize=(4 * 10, 2 * 10))

    ax[0, 0].axis('off')
    ax[0, 0].imshow(imgg)
    ax[0, 0].scatter(180, 200, s=1500, c='red', marker='o')  # 347

    plt.figure(figsize=(20, 20))
    plt.imshow(imgg)
    plt.scatter(180, 200, s=1500, c='red', marker='o')  # 683
    plt.axis('off')
    plt.savefig('subfigures/img_3.png', bbox_inches='tight', pad_inches=0)



    ax[0, 1].axis('off')
    wh = int(np.sqrt(attn_weights_sam.size(2)))
    attn_sam = attn_weights_sam[0][347].cpu().reshape(wh, wh)
    attn_sam = F.interpolate(attn_sam.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_sam = attn_sam / np.max(attn_sam)
    heatmap = cv2.applyColorMap((attn_sam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 1].imshow(blended)
    ax[0, 1].set_title("SAM", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/sam_3.png', bbox_inches='tight', pad_inches=0)

    ax[0, 2].axis('off')
    wh = int(np.sqrt(attn_weights_dino.size(2)))
    attn_dino = attn_weights_dino[0][347].cpu().reshape(wh, wh)
    attn_dino = F.interpolate(attn_dino.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                              align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dino = attn_dino / np.max(attn_dino)
    heatmap = cv2.applyColorMap((attn_dino * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 2].imshow(blended)
    ax[0, 2].set_title("DINO", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dino_3.png', bbox_inches='tight', pad_inches=0)

    ax[0, 3].axis('off')
    wh = int(np.sqrt(attn_weights_dinov2.size(2)))
    attn_dinov2 = attn_weights_dinov2[0][347].cpu().reshape(wh, wh)
    attn_dinov2 = F.interpolate(attn_dinov2.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dinov2 = attn_dinov2 / np.max(attn_dinov2)
    heatmap = cv2.applyColorMap((attn_dinov2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 3].imshow(blended)
    ax[0, 3].set_title("DINOV2", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dinov2_3.png', bbox_inches='tight', pad_inches=0)

    ax[1, 0].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    q_attnn = q_attn[0][347].cpu().reshape(wh, wh)
    q_attnn = F.interpolate(q_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    q_attnn = q_attnn / np.max(q_attnn)
    heatmap = cv2.applyColorMap((q_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 0].imshow(blended)
    ax[1, 0].set_title("Q_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qq_3.png', bbox_inches='tight', pad_inches=0)

    ax[1, 1].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    k_attnn = k_attn[0][347].cpu().reshape(wh, wh)
    k_attnn = F.interpolate(k_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    k_attnn = k_attnn / np.max(k_attnn)
    heatmap = cv2.applyColorMap((k_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 1].imshow(blended)
    ax[1, 1].set_title("K_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/kk_3.png', bbox_inches='tight', pad_inches=0)

    ax[1, 2].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    v_attnn = v_attn[0][347].cpu().reshape(wh, wh)
    v_attnn = F.interpolate(v_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    v_attnn = v_attnn / np.max(v_attnn)
    heatmap = cv2.applyColorMap((v_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 2].imshow(blended)
    ax[1, 2].set_title("V_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/vv_3.png', bbox_inches='tight', pad_inches=0)

    ax[1, 3].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    qk_attnn = qk_attn[0][347].cpu().reshape(wh, wh).detach()
    qk_attnn = F.interpolate(qk_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    qk_attnn = qk_attnn / np.max(qk_attnn)
    heatmap = cv2.applyColorMap((qk_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 3].imshow(blended)
    ax[1, 3].set_title("QK_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qk_3.png', bbox_inches='tight', pad_inches=0)

    plt.show()

    ########################################

    fig, ax = plt.subplots(2, 4, figsize=(4 * 10, 2 * 10))

    ax[0, 0].axis('off')
    ax[0, 0].imshow(imgg)
    ax[0, 0].scatter(400, 100, s=1500, c='red', marker='o')  # 192

    plt.figure(figsize=(20, 20))
    plt.imshow(imgg)
    plt.scatter(400, 100, s=1500, c='red', marker='o')  # 683
    plt.axis('off')
    plt.savefig('subfigures/img_4.png', bbox_inches='tight', pad_inches=0)

    ax[0, 1].axis('off')
    wh = int(np.sqrt(attn_weights_sam.size(2)))
    attn_sam = attn_weights_sam[0][192].cpu().reshape(wh, wh)
    attn_sam = F.interpolate(attn_sam.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_sam = attn_sam / np.max(attn_sam)
    heatmap = cv2.applyColorMap((attn_sam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 1].imshow(blended)
    ax[0, 1].set_title("SAM", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/sam_4.png', bbox_inches='tight', pad_inches=0)

    ax[0, 2].axis('off')
    wh = int(np.sqrt(attn_weights_dino.size(2)))
    attn_dino = attn_weights_dino[0][192].cpu().reshape(wh, wh)
    attn_dino = F.interpolate(attn_dino.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                              align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dino = attn_dino / np.max(attn_dino)
    heatmap = cv2.applyColorMap((attn_dino * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 2].imshow(blended)
    ax[0, 2].set_title("DINO", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dino_4.png', bbox_inches='tight', pad_inches=0)

    ax[0, 3].axis('off')
    wh = int(np.sqrt(attn_weights_dinov2.size(2)))
    attn_dinov2 = attn_weights_dinov2[0][192].cpu().reshape(wh, wh)
    attn_dinov2 = F.interpolate(attn_dinov2.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                                align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    attn_dinov2 = attn_dinov2 / np.max(attn_dinov2)
    heatmap = cv2.applyColorMap((attn_dinov2 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)

    ax[0, 3].imshow(blended)
    ax[0, 3].set_title("DINOV2", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/dinov2_4.png', bbox_inches='tight', pad_inches=0)

    ax[1, 0].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    q_attnn = q_attn[0][192].cpu().reshape(wh, wh)
    q_attnn = F.interpolate(q_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    q_attnn = q_attnn / np.max(q_attnn)
    heatmap = cv2.applyColorMap((q_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 0].imshow(blended)
    ax[1, 0].set_title("Q_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qq_4.png', bbox_inches='tight', pad_inches=0)

    ax[1, 1].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    k_attnn = k_attn[0][192].cpu().reshape(wh, wh)
    k_attnn = F.interpolate(k_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    k_attnn = k_attnn / np.max(k_attnn)
    heatmap = cv2.applyColorMap((k_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 1].imshow(blended)
    ax[1, 1].set_title("K_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/kk_4.png', bbox_inches='tight', pad_inches=0)

    ax[1, 2].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    v_attnn = v_attn[0][192].cpu().reshape(wh, wh)
    v_attnn = F.interpolate(v_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    v_attnn = v_attnn / np.max(v_attnn)
    heatmap = cv2.applyColorMap((v_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 2].imshow(blended)
    ax[1, 2].set_title("V_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/vv_4.png', bbox_inches='tight', pad_inches=0)

    ax[1, 3].axis('off')
    wh = int(np.sqrt(q_attn.size(2)))
    qk_attnn = qk_attn[0][192].cpu().reshape(wh, wh).detach()
    qk_attnn = F.interpolate(qk_attnn.unsqueeze(0).unsqueeze(0), size=(448, 448), mode='bilinear',
                             align_corners=False).squeeze(0).squeeze(0).detach().numpy()
    qk_attnn = qk_attnn / np.max(qk_attnn)
    heatmap = cv2.applyColorMap((qk_attnn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted((imgg.detach().numpy() * 255).astype(np.uint8), 0.5, heatmap, 0.8, 0)
    ax[1, 3].imshow(blended)
    ax[1, 3].set_title("QK_attn", fontsize=30)

    plt.figure(figsize=(20, 20))
    plt.imshow(blended)
    plt.axis('off')
    plt.savefig('subfigures/qk_4.png', bbox_inches='tight', pad_inches=0)

    plt.show()

    ########################################





