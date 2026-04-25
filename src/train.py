# ===============================================================
# ER + MixUp + RAR + Motion Novelty (RLVS) - Manual Baseline Selection
# MLflow Integrated Version
# ===============================================================
import os, random, math, time, json, copy, shutil
from collections import defaultdict
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from scipy.spatial.distance import cdist
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import mlflow.pytorch

# -------------------------
# EDITABLE: Choose ONE baseline
# -------------------------
BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/resnet18_violence_motion6ch.pth"
# BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/efficientnet_b0_6ch_best.pth"
# BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/mobilenet_v2_6ch_best.pth"
# BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/efficientnet_b4_violence_motion6ch_best.pth"
# BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/resnet34_violence_motion6ch_best.pth"
# BASE_MODEL_PATH = "/kaggle/input/datasets/hmzi67/models/resnet50_violence_motion6ch_best.pth"

VIDEO_ROOT = "/kaggle/input/datasets/mohamedmustafa/real-life-violence-situations-dataset/Real Life Violence Dataset"
CACHE_DIR  = "/kaggle/working/violence_motion_cache_for_cl"
OUT_DIR    = "/kaggle/working/er_motion_results"
MLFLOW_DIR = "/kaggle/working/mlruns"   # Kaggle local MLflow storage

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# -------------------------
# MLflow Setup
# -------------------------
# Option A: Local (recommended for Kaggle)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# Option B: EC2 remote (uncomment if EC2 is running and port 5000 open)
# mlflow.set_tracking_uri("http://YOUR_EC2_PUBLIC_IP:5000")

mlflow.set_experiment("rlvs-ocl-violence-detection")

# -------------------------
# args / hyperparams
# -------------------------
class_names = ["NonViolence", "Violence"]
num_classes  = len(class_names)

args = lambda: None
args.batch_size        = 10
args.buffer_batch_size = 10
args.mem_size          = 500
args.n_runs            = 5
args.subsample         = args.buffer_batch_size
args.lr                = 5e-5
args.epochs_per_task   = 5
args.replay_capacity   = args.mem_size * num_classes
args.mixup_alpha       = 0.1
args.d_eps             = 0.01
args.d_alpha           = 0.003
args.d_steps           = 3
args.d_coeff           = 1.0
args.device            = "cuda" if torch.cuda.is_available() else "cpu"
args.num_workers       = 2
args.seed              = 0
args.motion_weight     = 0.5

print("Device:", args.device)
print("Selected model:", os.path.basename(BASE_MODEL_PATH))

# -------------------------
# Motion Cache Builder
# -------------------------
def make_motion_cache(video_root, cache_dir, classes=class_names,
                      frames_per_video=16, debug=False):
    os.makedirs(cache_dir, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(cache_dir, cls), exist_ok=True)

    for cls in classes:
        folder = os.path.join(video_root, cls)
        if not os.path.isdir(folder):
            print(f"Warning: class folder not found: {folder}")
            continue
        files = sorted(glob(os.path.join(folder, "*.mp4")))
        if debug:
            print(f"{cls}: {len(files)} videos")

        for fp in tqdm(files, desc=f"Caching {cls}", leave=False):
            name       = os.path.basename(fp).replace(".mp4", ".pt")
            cache_path = os.path.join(cache_dir, cls, name)
            if os.path.exists(cache_path):
                continue

            cap          = cv2.VideoCapture(fp)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames       = []

            if total_frames <= 0:
                cap.release()
                blank  = np.zeros((224, 224, 3), dtype=np.uint8)
                tensor = torch.from_numpy(
                    np.concatenate([blank, blank], axis=2).astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                torch.save(tensor, cache_path)
                continue

            if total_frames < frames_per_video:
                indices = list(range(total_frames)) + [total_frames - 1] * (frames_per_video - total_frames)
            else:
                indices = np.linspace(0, total_frames - 1, frames_per_video).astype(int)

            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            cap.release()

            if len(frames) < 2:
                avg    = np.array(frames[0]) if frames else np.zeros((224, 224, 3), dtype=np.uint8)
                motion = np.zeros_like(avg)
            else:
                motion = np.zeros_like(frames[0], dtype=np.float32)
                for i in range(1, len(frames)):
                    diff    = cv2.absdiff(frames[i], frames[i - 1]).astype(np.float32)
                    motion += diff
                motion /= max(1, len(frames))
                motion  = motion.astype(np.uint8)
                avg     = np.mean(frames, axis=0).astype(np.uint8)

            stacked = np.concatenate([motion, avg], axis=2)   # motion first 0:3, RGB 3:6
            tensor  = torch.from_numpy(stacked.astype(np.float32) / 255.0).permute(2, 0, 1)
            torch.save(tensor, cache_path)

make_motion_cache(VIDEO_ROOT, CACHE_DIR, classes=class_names, frames_per_video=16, debug=False)

# -------------------------
# Dataset
# -------------------------
class CachedMotionDatasetFromCache(Dataset):
    def __init__(self, cache_dir, classes=class_names):
        self.paths, self.labels = [], []
        self.classes = classes
        for cls in classes:
            cls_folder = os.path.join(cache_dir, cls)
            if not os.path.isdir(cls_folder):
                continue
            for f in sorted(os.listdir(cls_folder)):
                if f.endswith(".pt"):
                    self.paths.append(os.path.join(cls_folder, f))
                    self.labels.append(classes.index(cls))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found under {cache_dir}")
        print(f"Loaded {len(self.paths)} cached tensors from {cache_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        t = torch.load(self.paths[idx], map_location='cpu')
        t = t.float()
        if t.ndim == 3:
            if t.shape[0] == 6:
                pass
            elif t.shape[0] == 3:
                t = torch.cat([t, t], dim=0)
            else:
                if t.shape[2] == 6:
                    t = t.permute(2, 0, 1)
                elif t.shape[2] == 3:
                    t = t.permute(2, 0, 1)
                else:
                    t = F.interpolate(t.unsqueeze(0), size=(6, 224, 224)).squeeze(0)
        if t.shape[1:] != (224, 224):
            t = F.interpolate(t.unsqueeze(0), size=(224, 224),
                              mode='bilinear', align_corners=False).squeeze(0)
        return t, self.labels[idx]

dataset = CachedMotionDatasetFromCache(CACHE_DIR, classes=class_names)

# -------------------------
# Motion Replay Buffer
# -------------------------
class MotionReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.store    = []
        self.n        = 0

    def add_batch(self, xb, yb):
        motion_mags = xb[:, 0:3, :, :].mean(dim=(1, 2, 3)).cpu().numpy()
        xb_cpu      = xb.detach().cpu()
        yb_cpu      = yb.detach().cpu()
        for x, y, m in zip(xb_cpu, yb_cpu, motion_mags):
            self.n += 1
            if len(self.store) < self.capacity:
                self.store.append((x.clone(), int(y), float(m)))
            else:
                j = random.randrange(self.n)
                if j < self.capacity:
                    self.store[j] = (x.clone(), int(y), float(m))

    def sample(self, k, exclude_labels=None, ret_ind=False):
        if not self.store:
            return (None, None) if not ret_ind else (None, None, None, None)

        inds = list(range(len(self.store)))
        if exclude_labels is not None:
            exclude_set = set(int(l.item()) for l in exclude_labels)
            inds        = [i for i in inds if self.store[i][1] not in exclude_set]
        if not inds:
            return (None, None) if not ret_ind else (None, None, None, None)

        k    = min(k, len(inds))
        mags = np.array([self.store[i][2] for i in inds]) + 1e-6

        motion_weights     = mags / mags.sum()
        labels             = np.array([self.store[i][1] for i in inds])
        class_weights      = np.zeros(len(inds))
        for cls in np.unique(labels):
            cls_mask               = labels == cls
            class_weights[cls_mask] = 1.0 / (np.sum(cls_mask) + 1e-6)
        class_weights_sum = class_weights.sum()
        if class_weights_sum > 0:
            class_weights /= class_weights_sum
        else:
            class_weights = np.ones(len(inds)) / len(inds)

        final_weights = args.motion_weight * motion_weights + (1 - args.motion_weight) * class_weights
        final_sum     = final_weights.sum()
        if final_sum == 0 or np.isnan(final_sum) or np.isinf(final_sum):
            final_weights = np.ones(len(inds)) / len(inds)
        else:
            final_weights /= final_sum

        chosen = np.random.choice(inds, k, p=final_weights, replace=False)
        xb     = torch.stack([self.store[i][0] for i in chosen]).to(args.device)
        yb     = torch.tensor([self.store[i][1] for i in chosen], device=args.device)

        if ret_ind:
            return xb, yb, chosen.tolist(), chosen.tolist()
        return xb, yb

# -------------------------
# Model Builder
# -------------------------
def make_model(num_classes=num_classes, base_weights=BASE_MODEL_PATH):
    detection_path = base_weights if base_weights is not None else BASE_MODEL_PATH
    backbone_name  = os.path.basename(detection_path).lower()

    if 'efficientnet_b4' in backbone_name or 'efficientnet-b4' in backbone_name:
        base = models.efficientnet_b4(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(6, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        base.features[0][0] = new_conv

        class EfficientNetB4WithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.features(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
        model = EfficientNetB4WithHidden(base)

    elif 'resnet50' in backbone_name:
        base    = models.resnet50(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(6, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride,
                               padding=old_conv.padding, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

        class ResNet50WithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.conv1(x); x = self.base.bn1(x)
                x = self.base.relu(x);  x = self.base.maxpool(x)
                x = self.base.layer1(x); x = self.base.layer2(x)
                x = self.base.layer3(x); x = self.base.layer4(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
        model = ResNet50WithHidden(base)

    elif 'resnet34' in backbone_name:
        base    = models.resnet34(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(6, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride,
                               padding=old_conv.padding, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

        class ResNet34WithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.conv1(x); x = self.base.bn1(x)
                x = self.base.relu(x);  x = self.base.maxpool(x)
                x = self.base.layer1(x); x = self.base.layer2(x)
                x = self.base.layer3(x); x = self.base.layer4(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
        model = ResNet34WithHidden(base)

    elif 'efficientnet' in backbone_name:
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(6, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        base.features[0][0] = new_conv

        class EfficientNetWithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.features(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
        model = EfficientNetWithHidden(base)

    elif 'mobilenet' in backbone_name:
        base = models.mobilenet_v2(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, num_classes)
        old_conv = base.features[0][0]
        new_conv = nn.Conv2d(6, old_conv.out_channels,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        base.features[0][0] = new_conv

        class MobileNetWithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.features(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                return torch.flatten(x, 1)
        model = MobileNetWithHidden(base)

    else:  # ResNet18 default
        base    = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(6, old_conv.out_channels,
                               kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride,
                               padding=old_conv.padding, bias=False)
        with torch.no_grad():
            base.conv1.weight[:, :3] = old_conv.weight
            base.conv1.weight[:, 3:] = old_conv.weight

        class ResNetWithHidden(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
            def forward(self, x):
                return self.base(x)
            def return_hidden(self, x):
                x = self.base.conv1(x); x = self.base.bn1(x)
                x = self.base.relu(x);  x = self.base.maxpool(x)
                x = self.base.layer1(x); x = self.base.layer2(x)
                x = self.base.layer3(x); x = self.base.layer4(x)
                x = self.base.avgpool(x)
                return torch.flatten(x, 1)
        model = ResNetWithHidden(base)

    if base_weights is not None and os.path.exists(base_weights):
        try:
            sd = torch.load(base_weights, map_location='cpu')
            model.load_state_dict(sd, strict=False)
            print(f"Loaded base weights from {base_weights} (strict=False).")
        except Exception as e:
            print("Warning loading base weights:", e)

    return model.to(args.device)

# -------------------------
# Helpers
# -------------------------
def mixup_batch(x1, y1, x2, y2, alpha=0.4):
    if x2 is None or len(x2) == 0:
        return None, None
    if len(x1) != len(x2):
        reps = math.ceil(len(x1) / len(x2))
        x2   = x2.repeat(reps, 1, 1, 1)[:len(x1)]
        y2   = y2.repeat(reps)[:len(x1)]
    lam   = float(np.random.beta(alpha, alpha))
    y1_oh = F.one_hot(y1, num_classes).float().to(x1.device)
    y2_oh = F.one_hot(y2, num_classes).float().to(x1.device)
    return lam * x1 + (1 - lam) * x2, lam * y1_oh + (1 - lam) * y2_oh

def soft_ce(pred, soft_y):
    return -(soft_y * F.log_softmax(pred, dim=1)).sum(dim=1).mean()

def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(args.device)
            out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())
    return accuracy_score(trues, preds) * 100 if trues else 0.0

def get_grad_vector_from_model(model):
    vecs = []
    for p in model.parameters():
        if p.grad is None:
            vecs.append(torch.zeros_like(p).view(-1))
        else:
            vecs.append(p.grad.view(-1).cpu())
    return torch.cat(vecs) if vecs else torch.zeros(0)

def apply_grad_to_copy(model, grad_vector, lr):
    mcopy       = copy.deepcopy(model)
    device      = next(mcopy.parameters()).device
    grad_vector = grad_vector.to(device)
    idx = 0
    for p in mcopy.parameters():
        n      = p.data.numel()
        g      = grad_vector[idx: idx + n].view_as(p.data)
        p.data = p.data - lr * g
        idx   += n
    return mcopy

def rar_targeted_linf(model_temp, model_for_logits, x_anchor, x_target,
                      eps=0.01, alpha=0.003, steps=3, rand_start=True):
    z = x_anchor.clone().detach()
    if rand_start:
        z = z + torch.empty_like(z).uniform_(-eps, eps)
    z = z.clamp(0, 1).detach().requires_grad_(True)
    with torch.no_grad():
        target_feat = model_temp.return_hidden(x_target).detach()
    for _ in range(steps):
        feat = model_temp.return_hidden(z)
        loss = ((feat - target_feat) ** 2).sum(dim=1).mean()
        loss.backward()
        grad    = z.grad.data
        z.data  = z.data - alpha * torch.sign(grad)
        z.data  = torch.max(torch.min(z.data, x_anchor + eps), x_anchor - eps)
        z.data  = z.data.clamp(0, 1)
        z.grad.zero_()
    return z.detach()

# -------------------------
# Build Tasks
# -------------------------
idx_by_class = defaultdict(list)
for i, lab in enumerate(dataset.labels):
    idx_by_class[lab].append(i)

tasks, names = [], []
for c_idx, cname in enumerate(class_names):
    idxs = idx_by_class.get(c_idx, [])
    if idxs:
        tasks.append(Subset(dataset, idxs))
        names.append(cname)

print("Tasks:", names, "| Sizes:", [len(t) for t in tasks])

# -------------------------
# Training Function — MLflow Integrated
# -------------------------
def run_variant(variant_name, run_number, base_weights=BASE_MODEL_PATH):
    print("\n" + "=" * 20, variant_name, "=" * 20)

    run_tags = {
        "dataset":        "RLVS",
        "backbone":       os.path.basename(BASE_MODEL_PATH).replace(".pth", ""),
        "variant":        variant_name,
        "run_number":     str(run_number),
        "novelty":        "motion_aware_reservoir_sampling",
        "motion_weight":  str(args.motion_weight),
    }

    with mlflow.start_run(
        run_name=f"{variant_name}_run{run_number}_{datetime.now().strftime('%H%M%S')}",
        tags=run_tags
    ):
        # ---------- Log all hyperparameters ----------
        mlflow.log_params({
            "variant":           variant_name,
            "backbone":          os.path.basename(BASE_MODEL_PATH),
            "batch_size":        args.batch_size,
            "buffer_batch_size": args.buffer_batch_size,
            "mem_size":          args.mem_size,
            "lr":                args.lr,
            "epochs_per_task":   args.epochs_per_task,
            "replay_capacity":   args.replay_capacity,
            "mixup_alpha":       args.mixup_alpha,
            "motion_weight":     args.motion_weight,
            "d_eps":             args.d_eps,
            "d_alpha":           args.d_alpha,
            "d_steps":           args.d_steps,
            "d_coeff":           args.d_coeff,
            "run_number":        run_number,
        })

        # ---------- Init model, optimizer, buffer ----------
        model        = make_model(num_classes, base_weights=base_weights)
        opt          = torch.optim.Adam(model.parameters(), lr=args.lr)
        buf          = MotionReplayBuffer(args.replay_capacity)
        history      = {n: [] for n in names}
        eval_loaders = [
            DataLoader(t, batch_size=32, shuffle=False, num_workers=args.num_workers)
            for t in tasks
        ]
        latencies    = []
        global_step  = 0

        # ---------- Task loop ----------
        for t_id, task in enumerate(tasks):
            print(f"\n--> Task {t_id + 1}/{len(tasks)}: {names[t_id]}")
            loader = DataLoader(task, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
            model.train()

            for epoch in range(args.epochs_per_task):
                epoch_losses = []

                for xb, yb in loader:
                    xb, yb     = xb.to(args.device), yb.to(args.device).long()
                    start_time = time.time()

                    # --- Standard ER step ---
                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()
                    latencies.append(time.time() - start_time)
                    epoch_losses.append(loss.item())
                    buf.add_batch(xb, yb)
                    global_step += 1

                    bx, by = buf.sample(args.buffer_batch_size)
                    if bx is not None:
                        opt.zero_grad()
                        F.cross_entropy(model(bx), by).backward()
                        opt.step()

                    # --- MixUp step ---
                    if "MixUp" in variant_name:
                        bx2, by2 = buf.sample(args.buffer_batch_size)
                        if bx2 is not None and len(bx2) > 0:
                            perm          = torch.randperm(len(bx2))
                            bx2_shuf, by2_shuf = bx2[perm], by2[perm]
                            mx, my        = mixup_batch(bx2, by2, bx2_shuf, by2_shuf,
                                                        alpha=args.mixup_alpha)
                            if mx is not None:
                                opt.zero_grad()
                                soft_ce(model(mx), my).backward()
                                opt.step()

                    # --- RAR step ---
                    if "RAR" in variant_name:
                        bx3, by3, subsample_inds, _ = buf.sample(
                            args.buffer_batch_size, exclude_labels=yb, ret_ind=True)
                        if bx3 is not None:
                            opt.zero_grad()
                            out2  = model(xb)
                            loss2 = F.cross_entropy(out2, yb)
                            loss2.backward()
                            grad_vec   = get_grad_vector_from_model(model)
                            model_temp = apply_grad_to_copy(model, grad_vec, lr=args.lr)
                            with torch.no_grad():
                                buf_feat   = model_temp.return_hidden(bx3).cpu().numpy()
                                batch_feat = model_temp.return_hidden(xb).cpu().numpy()
                            dist_mat   = cdist(buf_feat, batch_feat)
                            closest_idx = dist_mat.argmin(axis=1)
                            target_x   = xb[closest_idx, :]
                            linf_perturbed = rar_targeted_linf(
                                model_temp, model, bx3, target_x,
                                eps=args.d_eps, alpha=args.d_alpha,
                                steps=args.d_steps, rand_start=True)
                            loss_distilled = F.cross_entropy(model(linf_perturbed), by3)
                            loss_buffer    = F.cross_entropy(model(bx3), by3)
                            overall_loss   = ((1.0 - args.d_coeff) * loss_buffer
                                              + args.d_coeff * loss_distilled)
                            opt.zero_grad()
                            overall_loss.backward()
                            opt.step()

                # Log epoch-level loss to MLflow
                mlflow.log_metric("train_loss",
                                  np.mean(epoch_losses),
                                  step=global_step)

            # Evaluate after each task
            for j in range(t_id + 1):
                acc = evaluate(model, eval_loaders[j])
                history[names[j]].append(acc)
                mlflow.log_metric(f"acc_{names[j]}_after_task{t_id + 1}",
                                  acc, step=t_id + 1)

            print("Acc so far:",
                  [f"{a:.2f}" for a in [history[n][-1] for n in names[:t_id + 1]]])

        # ---------- Final metrics ----------
        forgets   = [max(history[n]) - history[n][-1] for n in names]
        avg_forget = np.mean(forgets)
        avg_acc    = np.mean([history[n][-1] for n in names])
        avg_latency = np.mean(latencies) * 1000 if latencies else 0.0

        nv_acc = history["NonViolence"][-1]
        v_acc  = history["Violence"][-1]
        nv_forget = forgets[0]
        v_forget  = forgets[1]

        mlflow.log_metrics({
            "avg_accuracy":           avg_acc,
            "avg_forgetting":         avg_forget,
            "nonviolence_accuracy":   nv_acc,
            "violence_accuracy":      v_acc,
            "nonviolence_forgetting": nv_forget,
            "violence_forgetting":    v_forget,
            "avg_latency_ms":         avg_latency,
        })

        # ---------- Save model + artifacts ----------
        variant_folder = os.path.join(OUT_DIR, variant_name.replace("+", "_"))
        os.makedirs(variant_folder, exist_ok=True)

        model_path = os.path.join(variant_folder, "model_final.pth")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        # Confusion matrix + classification report
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        y_true, y_pred = _predict_on_loader(model, test_loader)

        _save_confusion_matrix(y_true, y_pred, variant_folder, variant_name)
        _save_classification_report(y_true, y_pred, variant_folder)
        _save_fp_fn_csv(y_true, y_pred, variant_folder)

        mlflow.log_artifact(os.path.join(variant_folder, "confusion_matrix.png"))
        mlflow.log_artifact(os.path.join(variant_folder, "classification_report.txt"))
        mlflow.log_artifact(os.path.join(variant_folder, "FP_list.csv"))
        mlflow.log_artifact(os.path.join(variant_folder, "FN_list.csv"))

        # Count FP / FN
        fp_count = int(np.sum((np.array(y_pred) == 1) & (np.array(y_true) == 0)))
        fn_count = int(np.sum((np.array(y_pred) == 0) & (np.array(y_true) == 1)))
        mlflow.log_metrics({
            "false_positives": fp_count,
            "false_negatives": fn_count,
        })

        res = {
            "avg_acc":                  avg_acc,
            "avg_forget":               avg_forget,
            "per_task_acc":             [history[n][-1] for n in names],
            "per_task_forget":          forgets,
            "avg_latency_ms_per_batch": avg_latency,
            "false_positives":          fp_count,
            "false_negatives":          fn_count,
        }
        with open(os.path.join(variant_folder, "results.json"), "w") as f:
            json.dump(res, f, indent=2)
        mlflow.log_artifact(os.path.join(variant_folder, "results.json"))

        print(f"{variant_name} run{run_number} | "
              f"AvgAcc={avg_acc:.2f} | AvgForget={avg_forget:.2f} | "
              f"Latency={avg_latency:.2f}ms | FP={fp_count} | FN={fn_count}")

    return res

# -------------------------
# Helper: Predict
# -------------------------
def _predict_on_loader(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(args.device)
            out = model(xb)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            trues.extend(yb.numpy().tolist())
    return np.array(trues), np.array(preds)

# -------------------------
# Helper: Confusion Matrix
# -------------------------
def _save_confusion_matrix(y_true, y_pred, folder, variant_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.title(f"Confusion Matrix ({variant_name})")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "confusion_matrix.png"))
    plt.close()

# -------------------------
# Helper: Classification Report
# -------------------------
def _save_classification_report(y_true, y_pred, folder):
    with open(os.path.join(folder, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------
# Helper: FP / FN CSV
# -------------------------
def _save_fp_fn_csv(y_true, y_pred, folder):
    import csv
    fp_rows, fn_rows = [], []
    for i, (t, p) in enumerate(zip(y_true.tolist(), y_pred.tolist())):
        if p != t:
            if p == 1:
                fp_rows.append((i, class_names[t], class_names[p]))
            if p == 0 and t == 1:
                fn_rows.append((i, class_names[t], class_names[p]))

    for fname, rows in [("FP_list.csv", fp_rows), ("FN_list.csv", fn_rows)]:
        with open(os.path.join(folder, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index_in_test", "true_label", "pred_label"])
            for row in rows:
                w.writerow(row)

# -------------------------
# Multi-Run Over Variants
# -------------------------
variants         = ["ER", "ER+MixUp", "ER+RAR", "ER+MixUp+RAR"]
all_run_results  = {v: [] for v in variants}
run_acc_history  = {v: [] for v in variants}
run_forget_history = {v: [] for v in variants}

for run in range(args.n_runs):
    print(f"\n========== RUN {run + 1}/{args.n_runs} ==========")
    random.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(run)

    for v in variants:
        res = run_variant(v, run_number=run + 1)
        all_run_results[v].append(res)
        run_acc_history[v].append(res["avg_acc"])
        run_forget_history[v].append(res["avg_forget"])

# -------------------------
# Averaged Summary + MLflow Summary Run
# -------------------------
summary_avg = {}
for v in variants:
    accs           = [r["avg_acc"]    for r in all_run_results[v]]
    forgets        = [r["avg_forget"] for r in all_run_results[v]]
    per_task_acc   = np.array([r["per_task_acc"]    for r in all_run_results[v]])
    per_task_forget = np.array([r["per_task_forget"] for r in all_run_results[v]])
    summary_avg[v] = {
        "avg_acc_mean":        np.mean(accs),
        "avg_acc_std":         np.std(accs),
        "avg_forget_mean":     np.mean(forgets),
        "avg_forget_std":      np.std(forgets),
        "per_task_acc_mean":   np.mean(per_task_acc, axis=0),
        "per_task_acc_std":    np.std(per_task_acc, axis=0),
        "per_task_forget_mean": np.mean(per_task_forget, axis=0).tolist(),
    }

# Log summary as a separate MLflow run
with mlflow.start_run(run_name="SUMMARY_all_variants"):
    for v in variants:
        s        = summary_avg[v]
        v_safe   = v.replace("+", "_")
        mlflow.log_metrics({
            f"{v_safe}_avg_acc_mean":    s["avg_acc_mean"],
            f"{v_safe}_avg_acc_std":     s["avg_acc_std"],
            f"{v_safe}_avg_forget_mean": s["avg_forget_mean"],
            f"{v_safe}_avg_forget_std":  s["avg_forget_std"],
        })

print("\n\n================ FINAL SUMMARY ================")
for v in variants:
    s = summary_avg[v]
    print(f"\nVariant: {v}")
    print(f"AvgAcc    = {s['avg_acc_mean']:.2f} ± {s['avg_acc_std']:.2f}")
    print(f"AvgForget = {s['avg_forget_mean']:.2f} ± {s['avg_forget_std']:.2f}")
    for cname, m, sd in zip(class_names,
                             s['per_task_acc_mean'],
                             s['per_task_acc_std']):
        print(f"  {cname:12s}: {m:.2f} ± {sd:.2f}")
    print("Per-task forgetting:",
          [f"{x:.2f}" for x in s['per_task_forget_mean']])

# -------------------------
# Accuracy + Forgetting Plots
# -------------------------
plt.figure(figsize=(10, 6))
for v in variants:
    plt.plot(range(1, args.n_runs + 1), run_acc_history[v], label=v, marker='o')
plt.title("Average Accuracy over Runs (All Variants)")
plt.xlabel("Run"); plt.ylabel("Accuracy (%)")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "all_variants_accuracy.png"))
plt.close()
mlflow.log_artifact(os.path.join(OUT_DIR, "all_variants_accuracy.png"))

plt.figure(figsize=(10, 6))
for v in variants:
    plt.plot(range(1, args.n_runs + 1), run_forget_history[v], label=v, marker='o')
plt.title("Average Forgetting over Runs (All Variants)")
plt.xlabel("Run"); plt.ylabel("Forgetting (%)")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "all_variants_forgetting.png"))
plt.close()
mlflow.log_artifact(os.path.join(OUT_DIR, "all_variants_forgetting.png"))

# -------------------------
# Sample Visuals
# -------------------------
sample_folder = os.path.join(OUT_DIR, "sample_visuals")
os.makedirs(sample_folder, exist_ok=True)
indices = random.sample(range(len(dataset)), 3)
for i, idx in enumerate(indices):
    x, y  = dataset[idx]
    rgb   = (x[3:6].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    motion = (x[0:3].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).mean(
        axis=2, keepdims=True)
    motion = np.repeat(motion, 3, axis=2)
    cv2.imwrite(os.path.join(sample_folder, f"sample_{i}_rgb.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sample_folder, f"sample_{i}_motion.jpg"), motion)
mlflow.log_artifact(sample_folder)

# -------------------------
# ZIP output
# -------------------------
timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
final_zip_name = f"/kaggle/working/RLVS_OCL_MotionNovelty_Results_{timestamp}.zip"
shutil.make_archive(final_zip_name.replace(".zip", ""), 'zip', OUT_DIR)
print(f"\n[SUCCESS] ZIP created: {final_zip_name}")

# Also ZIP the mlruns folder so you can download and import on EC2
mlruns_zip = f"/kaggle/working/mlruns_export_{timestamp}.zip"
shutil.make_archive(mlruns_zip.replace(".zip", ""), 'zip', MLFLOW_DIR)
print(f"[SUCCESS] MLflow runs ZIP: {mlruns_zip}")
print("Download both ZIPs from Kaggle output section.")
print("All done!")