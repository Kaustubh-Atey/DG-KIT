import os
from DG_Dataset import DG_Dataset
from torchvision import transforms

root_dir = "/janaki/backup/users/student/rs/kaustubh.atey/DG_Datasets"
dataset_name = "OfficeHome"   # "PACS", "OfficeHome", "VLCS"

# (2) Build dataset (optionally pick a subset of domains)
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ds = DG_Dataset(
    root_dir=root_dir,
    dataset_name=dataset_name,
    domains=None,               # or e.g. ["Art", "Photo"]
    transform=tfm,
    return_domain=True,
)

# (3a) Plain DataLoader (no weighting)
loader_plain = ds.make_dataloader(batch_size=64, shuffle=True, weighted_sampling=False)
print("Plain batches:", len(loader_plain))

# (3b) Weighted by inverse CLASS frequency
loader_w_class = ds.make_dataloader(
    batch_size=64,
    weighted_sampling=True,
    weight_mode="class",        # options: "class", "domain", "class_in_domain"
    smoothing=0.1,              # optional stability
)
print("Weighted-by-class batches:", len(loader_w_class))

# (3c) Weighted by inverse (CLASS-IN-DOMAIN) frequency
loader_w_cid = ds.make_dataloader(
    batch_size=64,
    weighted_sampling=True,
    weight_mode="class_in_domain",
)
print("Weighted-by-class-in-domain batches:", len(loader_w_cid))

# (4) Iterate
imgs, y, d = next(iter(loader_w_class))
print("Batch shapes:", imgs.shape, y.shape, d.shape)

print(ds.class_to_idx, ds.domain_to_idx)
