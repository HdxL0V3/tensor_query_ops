import pyarrow as pa
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
import onnx

class Scan():
    def __init__(self, data):
        self.data = data
        self.index = 0

    def next(self):
        if self.index < len(self.data):
            result = self.data.slice(self.index, 1).to_pandas().iloc[0].to_numpy()
            result = result[np.newaxis, :]
            self.index += 1
            return result
        else:
            return None

class ProjectionModel(nn.Module):
    def __init__(self, proj):
        super(ProjectionModel, self).__init__()
        self.proj = proj

    def forward(self, x):
        return x[:, self.proj]

class Projection():
    def __init__(self, child, columns):
        self.child = child
        self.columns = columns

    def next(self):
        tensor = self.child.next()
        if tensor is None:
            return None
        tensor = tensor.astype(np.float32)
        model = ProjectionModel(self.columns)
        # 将模型转换为ONNX格式
        dummy_input = torch.randn(tensor.shape[0], tensor.shape[1])
        torch.onnx.export(model, dummy_input, "projection_model.onnx", input_names=["input"], output_names=["output"], verbose=False)
        # 使用ONNX Runtime执行模型
        ort_session = ort.InferenceSession("projection_model.onnx")
        # 运行模型并获取输出
        ort_inputs = {"input": tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        # 输出结果
        result = ort_outputs[0]
        return result

class SelectionModel(nn.Module):
    def __init__(self, predicate):
        super(SelectionModel, self).__init__()
        self.predicate = predicate

    def forward(self, x):
        mask = self.predicate(x)
        # print("mask")
        return mask

class Selection():
    def __init__(self, child, predicate):
        self.child = child
        self.predicate = predicate

    def next(self):
        while True:
            tensor = self.child.next()
            if tensor is None:
                return None
            tensor = tensor.astype(np.float32)
            model = SelectionModel(self.predicate)
            dummy_input = torch.randn(tensor.shape[0], tensor.shape[1])
            torch.onnx.export(model, dummy_input, "selection_model.onnx", input_names=["input"], output_names=["output"], verbose=False)
            ort_session = ort.InferenceSession("selection_model.onnx")
            ort_inputs = {"input": tensor}
            ort_outputs = ort_session.run(None, ort_inputs)
            # print(ort_outputs[0][0])
            if ort_outputs[0][0]:
                return tensor

# 示例数据
data = [
    {'id': 1, 'gender': 0, 'age': 38, 'length': 177.2},
    {'id': 2, 'gender': 1, 'age': 40, 'length': 178.8},
    {'id': 3, 'gender': 0, 'age': 35, 'length': 175.5}
]
data0 = pa.Table.from_pylist(data)
# 构建查询计划
scan0 = Scan(data0)
# print(scan0.next())
projection0 = Projection(scan0, [0, 2, 3])
selection0 = Selection(projection0, lambda t: t[:, 1] < 39)
# selection0 = Selection(projection0, lambda t: t[:, 2] == 177.2)

# 执行查询计划
while True:
    tuple = selection0.next()
    if tuple is None:
        break
    print(tuple)

