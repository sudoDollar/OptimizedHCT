from hct_base import HCTBase
from data import LIU4K
from inference import InferenceEngine
import torch
import os

def get_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    s = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return s

def print_size_of_model(model):
    print('Size (MB):', get_model_size(model))

def evaluate_batch(ie, data_loader):
    corr = 0
    tot = 0
    
    for num, batch in enumerate(data_loader):
        print(f"{num+1} / {len(data_loader)}", end="\r")
        pred, exec_time = ie.predict_batch(batch[0])
        
        corr_labels = list(map(lambda i: ie.labelMap[int(i)], batch[1]))
        tot += len(corr_labels)
        
        for i in range(len(corr_labels)):
            corr += corr_labels[i] == pred[i]

    # return acc
    return corr / tot, corr, tot


hct = HCTBase()
batch_size = 16
val_dataset = LIU4K("../dataset/image_labels_valid.csv", False)

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")

# print("Evalulating unquantized model")
# acc_norm, corr_norm, tot_norm = evaluate_batch(ie, val_loader)
# print(f"Accuracy before Quantization = {acc_norm}; {corr_norm}/{tot_norm}")

# quantize model
print("Quantizing model (dynamic)")
ie.quantize()

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# load quantized model
# print("Evalulating quantized model (dynamic)")
# acc_q, corr_q, tot_q = evaluate_batch(ie, val_loader)
# print(f"Accuracy after dynamic Quantization = {acc_q}; {corr_q}/{tot_q}")


# check inference times
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
it = iter(val_loader)
data = next(it)

print("Per Batch")

# Eager Model
ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
pred, time = ie.predict_batch(data[0])
print("Eager Model Inference (CPU)")
print(pred)
print("Inference Time per Batch: {} secs\n".format(time))
print_size_of_model(ie.model)

ie.quantize() 

# Quantized Model
pred, time = ie.predict_batch(data[0])
print("Quantized (dynamic) Model Inference (CPU)")
print(pred)
print("Inference Time per Batch: {} secs\n".format(time))
print_size_of_model(ie.model)

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
# Graph Model (Torchscript)
print("Graph Model Inference (CPU)")
ie.compile(output_fn="torchscript_hct_1.pth")
pred, time = ie.predict_batch(data[0])
print("Inference Time per Batch: {} secs\n".format(time))

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
ie.quantize() 

# Graph Model (Torchscript) + Quantized (dynamic)
print("Graph + Quantized (dynamic) Model Inference (CPU)")
ie.compile("torchscript_quantized_dynamic_hct.pth")
pred, time = ie.predict_batch(data[0])
print("Inference Time per Batch: {} secs\n".format(time))


print("Per Image")

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
it = iter(val_loader)
data = next(it)

# Eager Model
pred, time = ie.predict(data[0])
print("Eager Model Inference (CPU)")
print(pred)
print("Inference Time per Image: {} secs\n".format(time))
print_size_of_model(ie.model)

ie.quantize()

# Quantized Model
pred, time = ie.predict(data[0])
print("Quantized Model Inference (CPU)")
print(pred)
print("Inference Time per Image: {} secs\n".format(time))
print_size_of_model(ie.model)

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
# Graph Model (Torchscript)
print("Graph Model Inference (CPU)")
ie.compile(output_fn="torchscript_hct_1.pth")
pred, time = ie.predict(data[0])
print("Inference Time per Image: {} secs\n".format(time))

ie = InferenceEngine(hct, "../saved_model/hct.pt", "cpu")
ie.quantize()

# Graph Model (Torchscript) + Quantized
print("Graph + Quantized Model Inference (CPU)")
ie.compile("torchscript_quantized_hct.pth")
pred, time = ie.predict(data[0])
print("Inference Time per Image: {} secs\n".format(time))