from copy import deepcopy
import torch
import torch.nn as nn
import torch.ao.quantization
import os
import time

class InferenceEngine:

    labelMap = {
        0: "Animal",
        1: "Building",
        2: "Mountain",
        3: "Street"
    }

    def __init__(self, model:nn.Module, model_state_path:str, device:str, quantized=False) -> None:
        self.model = deepcopy(model)
        self.device = torch.device(device)
        self.compiled = False
        self.quantized = False
        
        if quantized:
            backend = "x86"
            self.model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
            torch.backends.quantized.engine = backend
            torch.ao.quantization.prepare(self.model, inplace=True)
            torch.ao.quantization.convert(self.model, inplace=True)
            self.quantized = True

        self.model.load_state_dict(torch.load(model_state_path, map_location=self.device))
        self.model_state_path = model_state_path
        self.model_state_dir = os.path.dirname(self.model_state_path)
        self._model = model
        self.model.eval()
        self.model.to(self.device)

    def compile(self, output_fn="torchscript_hct.pth"):
        """
        Trace the Module using torch.jit.trace for better Inference performance
        """
        # if self.quantized:
        #     print("Currently, Tracing/Scripting quantized model not supported")
        #     return
                    
        inputs = (torch.rand(1, 3, 512, 512).to(self.device),)
        check_inputs = [(torch.rand(1, 3, 520, 520).to(self.device),), (torch.rand(1, 3, 612, 612).to(self.device),)]
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, inputs, check_inputs=check_inputs)
            self.scripted_model_state_path = os.path.join(self.model_state_dir, output_fn)
            traced_model.save(self.scripted_model_state_path)
        
        self.compiled = True
        print("Model Successfully compiled/traced to {}".format(self.scripted_model_state_path))
        print("Further predictions will be made using traced module")

        self.model = torch.jit.load(self.scripted_model_state_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def quantize(self, data_loader, output_fn="ptq_hct.pth"):
        """
        Only for CPU
        """
        if self.quantized:
            print("Model already quantized")
            return
        if self.compiled:
            print("Currently, quantizing Scripted model not supported")
            return
        if str(self.device) == "cuda":
            print("ONLY CPU Supported")
            return
        backend = "x86"
        self.model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend
        torch.ao.quantization.prepare(self.model, inplace=True)
        self.calibrate(data_loader)
        torch.ao.quantization.convert(self.model, inplace=True)

        self.quantized_model_state_path = os.path.join(self.model_state_dir, output_fn)
        torch.save(self.model.state_dict(), self.quantized_model_state_path)
        self.model.load_state_dict(torch.load(self.quantized_model_state_path))
        self.model.eval()
        self.quantized = True

    def calibrate(self, data_loader):
        with torch.no_grad():
            counter = 0
            for data in data_loader:
                counter += 1 # DEBUG
                print(f"{counter} / {len(data_loader)}", end="\r") # DEBUG
                
                x = data[0].to(self.device)
                y = data[1].to(self.device)
                self.model(x)

    def predict(self, x:torch.Tensor) -> str:
        """
        Predict Single Image
        """
        with torch.no_grad():
            if str(self.device) == "cuda": 
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            X = x.to(self.device)
            y_pred = self.model(X)
            top_pred = y_pred.argmax(1, keepdim =True)
            if str(self.device) == "cuda": 
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            return self.labelMap[top_pred.item()], end_time - start_time
        
    def predict_batch(self, x:torch.Tensor) -> list:
        """
        Predict Whole batch of data
        """
        with torch.no_grad():
            if str(self.device) == "cuda": 
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            X = x.to(self.device)
            y_pred = self.model(X)
            top_pred = y_pred.argmax(1, keepdim =True)
            pred = top_pred.squeeze().tolist()
            pred = list(map(lambda x: self.labelMap[x], pred))
            if str(self.device) == "cuda": 
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            return pred, end_time - start_time

    def reset(self, device="cuda"):
        self.model = deepcopy(self._model)
        self.device = torch.device(device)
        self.model.load_state_dict(torch.load(self.model_state_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.compiled = False
        self.quantized = False
        self.quantized_model_state_path = None
        self.scripted_model_state_path = None