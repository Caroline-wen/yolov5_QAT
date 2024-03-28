import torch
import re
import yaml
import json 
import os  
import collections
from pathlib import Path
from pytorch_quantization import quant_modules
from yolov5.models.yolo import Model
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
from copy import deepcopy
from yolov5.utils.dataloaders import create_dataloader
import yolov5.val as val


def load_yolov5_model(weight, device='cpu'):
    ckpt = torch.load(weight, map_location=device)
    model = Model('yolov5/models/yolov5s.yaml', ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model


# input: Max ==> Histogram
def initialize():
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    
    quant_logging.set_verbosity(quant_logging.ERROR)


def prepare_model(weight, device):  # 自动为YOLOv5模型插入量化节点
    # quant_modules.initialize()  # 自动量化采用Max确定scale
    initialize()
    model = load_yolov5_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()  # conv bn进行层的合并, 提高模型推理速度
    return model



def transfer_torch_to_quantization(nn_instance, quant_module):
    quant_instance = quant_module.__new__(quant_module)
    
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)
    
    def __init__(self):
        # 返回两个QuantDesriptor的实例  self.__class__ 是 quant_instance的类, EX, QuantConv2d
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True  # 采用直方图法, 加速量化
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True
        
    __init__(quant_instance)
    return quant_instance


def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False


# 递归
def torch_module_find_quant_module(module, module_dict, ignore_layer, prefix=''):
    for name in module._modules:
        submodule = module._modules[name]
        path = name if prefix =='' else prefix + '.' + name
        torch_module_find_quant_module(submodule, module_dict, ignore_layer, prefix=path)

        submodule_id = id(type(submodule))
        if submodule_id in module_dict:
            ignored = quantization_ignore_match(ignore_layer, path)  # 判断当前层是否是被忽略的层
            if ignored:
                print(f"Quantization: {path} has ignored. ")
                continue
            # 转换
            module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])


def replace_to_quantization_model(model, ignore_layer=None):  # 手动量化
    module_dict = {}
    
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod
    
    torch_module_find_quant_module(model, module_dict, ignore_layer)


def prepare_val_dataset(cocodir, batch_size=4):
    dataloader = create_dataloader(
        f"{cocodir}/images/train2017",
        imgsz=640,
        batch_size=batch_size,
        augment=False, hyp=None, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader


def prepare_train_dataset(cocodir, batch_size=4):
    
    with open("yolov5/data/hyps/hyp.scratch-low.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # 加载训练时参数
    
    # 需要加载yaml的参数时, 修改为augment=True, hyp=hyp
    dataloader = create_dataloader(
        f"{cocodir}/images/train2017",
        imgsz=640,
        batch_size=batch_size,
        augment=True, hyp=hyp, rect=True, cache=False, stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloader

from yolov5.utils.general import init_seeds, check_dataset
def evaluate_coco(model, loader, save_dir='.', conf_threshold=0.001, iou_threshold=0.65):
    
    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    model = deepcopy(model)
    return val.run(
        check_dataset("./yolov5/data/coco128.yaml"),
        save_dir=Path(save_dir),
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        model=model,
        dataloader=loader,
        plots=False,  # 是否生成图像
        save_json=False  # 是否将结果保存成json
    )[0][3]


def collect_stats(model, data_loader, device, num_batch=200):
    model.eval()
    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    # test
    with torch.no_grad():
        for i, datas in enumerate(data_loader):
            imgs = datas[0].to(device, non_blocking=True).float()/255.0
            model(imgs)
            
            if i >= num_batch:
                break
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, device, **kwargs):
    
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):  # 如果是Max校准器, 使用默认参数
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)  # 如果不是Max校准器, 使用传递的参数
                module._amax = module._amax.to(device)


class disable_quantization():  # 关闭量化
    def __init__(self, model):
        self.model = model
    
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            module._disabled = disabled
    
    def __enter__(self):
        self.apply(disabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)


class enable_quantization():  # 重启量化
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            module._disabled = not enabled
    
    def __enter__(self):
        self.apply(enabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)


def calibrate_model(model, dataloader, device):
    # 收集量化信息
    collect_stats(model, dataloader, device)
    # 计算动态范围, 计算amax值, scale
    compute_amax(model, device, method='mse')  # 第二个参数是histogram计算amax值的方式: [entropy, mse, percentile] 相对熵法 均方误差 百分比


# 判断层是否是量化层
def have_quantizer(layer):
    for name, module in layer.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True


import json
class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []
    
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)  # indent使缩进值


def sensitive_analysis(model, loader, save_file):
    # save_file = "sensitive_analysis.json"
    
    summary = SummaryTool(save_file)  # 用于保存精度值
    # for 循环每一个层
    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断layer是否是量化层
        if have_quantizer(layer):  # 如果是量化层
            # 使该层的量化失效, 不进行int8的量化, 使用fp16进行运算
            # disable_quantization(layer).apply()
            with disable_quantization(layer):
                # 计算mAP值
                ap = evaluate_coco(model, loader)
            # 保存精度值, json文件
            summary.append([ap, f"model.{i}"])
            print(f"layer {i} ap: {ap}") 
            # 重启该层的量化, 还原
            # enable_quantization(layer).apply()
        else:
            print(f"ignore model. {i} because it is {type(layer)}")
    # 循环结束, 打印前10个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary: ")
    for i, (ap, name) in enumerate(summary[:10]):
        print(f"Top {i}: using fp16  {name}, ap = {ap:.5f}")
            


def export_ptq(model, save_file, device, dynamic_batch=False):
    
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    model.eval()
    
    with torch.no_grad():
        torch.onnx.export(model, input_dummy, save_file, opset_version=13, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None,
                        )
    quant_nn.TensorQuantizer.use_fb_fake_quant = False


class QuantAdd(torch.nn.Module):
    def __init__(self, quantization):
        super().__init__()

        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, calib_method="histogram"))
            self._input0_quantizer._calibrator._torch_hist = True
            self._input1_quantizer._calibrator._torch_hist = True
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y

def bottleneck_quant_forward(self, x):
    if hasattr(self, "addop"):
        return self.addop(x, self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)))
    else:
        return x + self.cv2(self.cv1(x) if self.add else self.cv2(self.cv1(x)))

def replace_bottleneck_forward(model):
    for name, bottleneck in model.named_modules():
        if bottleneck.__class__.__name__ == "Bottleneck":
            if bottleneck.add:
                if not hasattr(bottleneck, "addop"):
                    print(f"Add QuantAdd to {name}")
                    bottleneck.addop = QuantAdd(bottleneck.add)
                bottleneck.__class__.forward = bottleneck_quant_forward