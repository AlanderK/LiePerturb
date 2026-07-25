from .clip import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blip_models.blip import blip_decoder


class StandardMLPAdapter(nn.Module):
    def __init__(self, dim, hidden_dim, scale):
        super().__init__()
        self.scale = scale
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim * 2)
        )

    def forward(self, img_features, text_features):
        delta_img, delta_text = self.mlp(torch.cat([img_features, text_features], dim=1)).chunk(2, dim=1)
        return img_features + self.scale * delta_img, text_features + self.scale * delta_text


class NormalizedMLPAdapter(StandardMLPAdapter):
    def forward(self, img_features, text_features):
        img_norm = F.normalize(img_features, dim=1)
        text_norm = F.normalize(text_features, dim=1)
        delta_img, delta_text = self.mlp(torch.cat([img_norm, text_norm], dim=1)).chunk(2, dim=1)
        img_out = F.normalize(img_norm + self.scale * delta_img, dim=1)
        text_out = F.normalize(text_norm + self.scale * delta_text, dim=1)
        return img_out, text_out


class LoRAAdapter(nn.Module):
    def __init__(self, dim, rank, scale):
        super().__init__()
        self.scale = scale
        self.down = nn.Linear(dim * 2, rank, bias=False)
        self.up_img = nn.Linear(rank, dim, bias=False)
        self.up_text = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up_img.weight)
        nn.init.zeros_(self.up_text.weight)

    def forward(self, img_features, text_features):
        low_rank = self.down(torch.cat([img_features, text_features], dim=1))
        img_delta = self.up_img(low_rank)
        text_delta = self.up_text(low_rank)
        return img_features + self.scale * img_delta, text_features + self.scale * text_delta


class OrthogonalLinearAdapter(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.scale = scale
        self.img_projection = nn.Parameter(torch.empty(dim, dim))
        self.text_projection = nn.Parameter(torch.empty(dim, dim))
        self.gate = nn.Linear(dim * 2, dim * 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.img_projection)
        nn.init.orthogonal_(self.text_projection)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, img_features, text_features):
        gates = torch.sigmoid(self.gate(torch.cat([img_features, text_features], dim=1)))
        img_gate, text_gate = gates.chunk(2, dim=1)
        img_cross = text_features @ self.text_projection.t()
        text_cross = img_features @ self.img_projection.t()
        return img_features + self.scale * img_gate * img_cross, text_features + self.scale * text_gate * text_cross

    def reorthogonalize(self):
        with torch.no_grad():
            for projection in (self.img_projection, self.text_projection):
                u, _, vh = torch.linalg.svd(projection.data.float(), full_matrices=False)
                projection.copy_((u @ vh).to(dtype=projection.dtype, device=projection.device))


class CalibratedCosineHead(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        alpha_min: float = 1.0,
        alpha_max: float = 50.0,
        alpha0: float = 10.0,
        delta0: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if alpha_min <= 0 or alpha_max <= alpha_min:
            raise ValueError("alpha range must satisfy 0 < alpha_min < alpha_max.")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eps = eps
        self.raw_alpha = nn.Parameter(torch.empty(()))
        self.delta_net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.set_calibration(alpha0, delta0)

    @property
    def alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)

    def set_calibration(self, alpha0: float, delta0: float):
        alpha0 = float(min(max(alpha0, self.alpha_min + 1e-4), self.alpha_max - 1e-4))
        delta0 = float(min(max(delta0, -0.99), 0.99))
        alpha_ratio = (alpha0 - self.alpha_min) / (self.alpha_max - self.alpha_min)
        alpha_ratio = min(max(alpha_ratio, 1e-6), 1.0 - 1e-6)
        raw_alpha = torch.logit(torch.tensor(alpha_ratio, dtype=torch.float32))
        raw_delta = torch.atanh(torch.tensor(delta0, dtype=torch.float32))
        with torch.no_grad():
            self.raw_alpha.copy_(raw_alpha.to(device=self.raw_alpha.device, dtype=self.raw_alpha.dtype))
            final_layer = self.delta_net[-1]
            final_layer.weight.zero_()
            final_layer.bias.fill_(raw_delta.to(device=final_layer.bias.device, dtype=final_layer.bias.dtype).item())

    def forward(self, img_feat, txt_feat):
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        score = F.cosine_similarity(img_feat, txt_feat, dim=-1, eps=self.eps)
        alpha = self.alpha
        delta = torch.tanh(self.delta_net(torch.cat([img_feat, txt_feat], dim=-1)).squeeze(-1))
        logits = alpha * (score - delta)
        probabilities = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "score": score,
            "alpha": alpha,
            "delta": delta,
        }


class PerturbModel(nn.Module):
    def __init__(self, opt):
        super(PerturbModel, self).__init__()
        self.opt = opt
        self.cross_modal_mode = opt.cross_modal_mode.lower()
        self.use_fam_loss = self.cross_modal_mode == "fam"

        clip_name = opt.arch[5:] if opt.arch.startswith("CLIP:") else opt.arch
        if clip_name != "ViT-L/14":
            raise ValueError("Cross-modal semantic consistency training only supports CLIP:ViT-L/14.")

        self.model1, self.preprocess = clip.load(clip_name, device="cpu")
        if not hasattr(self.model1, "encode_image_text") or self.model1.fam is None:
            raise RuntimeError("Loaded CLIP model does not have an enabled FAM module.")

        self.model2 = blip_decoder(
            pretrained=opt.blip_pretrained,
            image_size=opt.blip_image_size,
            vit=opt.blip_vit
        )

        self.model1.eval()
        self.model2.eval()
        for _, param in self.model1.named_parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

        dim = opt.cross_modal_dim
        clip_dim = self.model1.text_projection.shape[1]
        if dim != clip_dim:
            raise ValueError(f"cross_modal_dim={dim} does not match CLIP output dim {clip_dim}.")
        hidden_dim = int(dim * opt.adapter_hidden_ratio)
        self.adapter = self._build_adapter(self.cross_modal_mode, dim, hidden_dim, opt)
        self.classifier = CalibratedCosineHead(dim, hidden_dim)
        if self.use_fam_loss:
            for param in self.model1.fam.parameters():
                param.requires_grad = True

    @staticmethod
    def _build_adapter(mode, dim, hidden_dim, opt):
        if mode == "fam":
            return nn.Identity()
        if mode == "mlp":
            return StandardMLPAdapter(dim, hidden_dim, opt.adapter_scale)
        if mode == "norm_mlp":
            return NormalizedMLPAdapter(dim, hidden_dim, opt.adapter_scale)
        if mode == "lora":
            return LoRAAdapter(dim, opt.lora_rank, opt.adapter_scale)
        if mode == "orthogonal":
            return OrthogonalLinearAdapter(dim, opt.adapter_scale)
        raise ValueError("cross_modal_mode should be one of [fam, mlp, norm_mlp, lora, orthogonal].")

    def train(self, mode: bool = True):
        super().train(mode)
        self.model1.eval()
        self.model2.eval()
        if self.use_fam_loss:
            self.model1.fam.train(mode)
        else:
            self.model1.fam.eval()
        self.adapter.train(mode)
        return self

    def get_caption(self, model, img):
        with torch.no_grad():
            return model.generate(
                img,
                sample=True,
                top_p=0,
                max_length=30,
                min_length=5
            )

    def _encode_frozen_clip(self, x, text):
        with torch.no_grad():
            img_features = self.model1.encode_image(x)
            text_features = self.model1.encode_text(text)
        return img_features.float(), text_features.float()

    def forward_features(self, x):
        # BLIP can occasionally generate a degenerate repeated caption whose
        # tokenized length exceeds CLIP's fixed 77-token context.  Keep the
        # start of the generated caption and preserve the end-of-text token
        # instead of aborting an otherwise valid image batch.
        text = clip.tokenize(
            self.get_caption(self.model2, x), truncate=True
        ).to(x.device)
        if self.use_fam_loss:
            img_features, text_features = self.model1.encode_image_text(x, text)
        else:
            img_features, text_features = self._encode_frozen_clip(x, text)
            img_features, text_features = self.adapter(img_features, text_features)

        head_outputs = self.classifier(img_features.float(), text_features.float())
        return img_features, text_features, head_outputs

    def forward_outputs(self, x):
        return self.forward_features(x)[2]

    def forward(self, x):
        return self.forward_outputs(x)["probabilities"]

    def orthogonal_loss(self):
        if not self.use_fam_loss:
            return next(self.parameters()).new_tensor(0.0)
        return self.model1.fam.orthogonal_loss()

    def reorthogonalize_fam(self):
        if self.use_fam_loss:
            self.model1.fam.reorthogonalize_projections()


def initialize_from_training_loader(
    model,
    train_loader,
    device,
    q=0.65,
    max_batches=4,
):
    if max_batches is None or max_batches <= 0:
        raise ValueError("max_batches must be a positive integer for cosine head initialization.")
    was_training = model.training
    real_scores = []
    fake_scores = []
    used_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            outputs = model.forward_outputs(img.to(device))
            score = outputs["score"].detach().float().cpu()
            label = label.view(-1).cpu()
            real_scores.append(score[label == 0])
            fake_scores.append(score[label == 1])
            used_batches += 1

    real_scores = [scores for scores in real_scores if scores.numel() > 0]
    fake_scores = [scores for scores in fake_scores if scores.numel() > 0]
    if len(real_scores) == 0 or len(fake_scores) == 0:
        raise RuntimeError(
            "Cannot initialize cosine head: sampled {} batch(es) did not contain both real and fake samples. "
            "Increase --cosine_head_init_batches.".format(used_batches)
        )

    real_scores = torch.cat(real_scores)
    fake_scores = torch.cat(fake_scores)
    mu_real = real_scores.mean().item()
    mu_fake = fake_scores.mean().item()
    separation = mu_real - mu_fake
    delta0 = (mu_real + mu_fake) / 2.0

    if separation <= 0:
        print(
            "Warning: raw cosine direction is not aligned with real probability "
            "(mu_real <= mu_fake). Keeping alpha positive and using conservative alpha0=5.0."
        )
        alpha0 = 5
    else:
        logit_q = torch.logit(torch.tensor(q)).item()
        alpha0 = 2.0 * logit_q / (separation + model.classifier.eps)
        alpha0 = min(max(alpha0, 1.05), 14.30)

    model.classifier.set_calibration(alpha0, delta0)
    if was_training:
        model.train()

    print(
        "Initialized calibrated cosine head: "
        "used_batches={}, mu_real={:.6f}, mu_fake={:.6f}, separation={:.6f}, "
        "alpha0={:.6f}, delta0={:.6f}".format(
            used_batches, mu_real, mu_fake, separation, alpha0, delta0
        )
    )
    return {
        "used_batches": used_batches,
        "mu_real": mu_real,
        "mu_fake": mu_fake,
        "separation": separation,
        "alpha0": alpha0,
        "delta0": delta0,
    }
