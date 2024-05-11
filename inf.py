from data import LIU4K
from hct_base import HCTBase
from params import Config
from inference import InferenceEngine
import torch
import time

config = Config().parse()
hct = HCTBase()

#Data Loader
val_dataset = LIU4K("dataset/image_labels_valid.csv", False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
it = iter(val_loader)
data = next(it)

# Inference Analysis (CPU)
# We will use same image for both eager and graph mode to compare the results
ie = InferenceEngine(hct, "saved_model/hct.pt", "cpu")

# Eager Model
pred, time = ie.predict_batch(data[0])
print("Eager Model Inference (CPU)")
print(pred)
print("Inference Time per Batch: {} secs\n".format(time))

# Graph Model (Torchscript)
ie.compile()

pred, time = ie.predict_batch(data[0])
print("Graph Model Inference (CPU)")
print(pred)
print("Inference Time per Batch: {} secs\n".format(time))

# Inference Analysis (GPU)
# We will use same image for both eager and graph mode to compare the results
ie = InferenceEngine(hct, "saved_model/hct.pt", "cuda")

# Eager Model
#WarmUp
print("Eager Model Inference (GPU)")
print(ie.predict_batch(data[0])[0])

pred, time = ie.predict_batch(data[0])
print("Inference Time per Batch: {} secs\n".format(time))

# Graph Model (Torchscript)
ie.compile()
#WarmUp
print("Graph Model Inference (GPU)")
print(ie.predict_batch(data[0])[0])

pred, time = ie.predict_batch(data[0])
print("Inference Time per Batch: {} secs\n".format(time))






