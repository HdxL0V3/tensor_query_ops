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

class JoinModel(nn.Module):
    def __init__(self, left_key, right_key):
        super(JoinModel, self).__init__()
        self.left_key = left_key
        self.right_key = right_key

    def forward(self, left_data, right_data):
        left_key_data = left_data[:, self.left_key]
        right_key_data = right_data[:, self.right_key]
        
        result = []
        for left_row in left_data:
            left_key_value = left_row[self.left_key]
            for right_row in right_data:
                right_key_value = right_row[self.right_key]
                if left_key_value == right_key_value:
                    joined_row = np.concatenate((left_row, right_row))
                    result.append(joined_row)
        
        if len(result) > 0:
            result = np.stack(result)
        else:
            result = np.empty((0, left_data.shape[1] + right_data.shape[1]))
        
        return result

class Join():
    def __init__(self, left_child, right_child, left_key, right_key):
        self.left_child = left_child
        self.right_child = right_child
        self.left_key = left_key
        self.right_key = right_key

    def next(self):
        left_data = self.left_child.next()
        right_data = self.right_child.next()

        if left_data is None or right_data is None:
            return None

        left_data = left_data.astype(np.float32)
        right_data = right_data.astype(np.float32)

        model = JoinModel(self.left_key, self.right_key)
        dummy_left_input = torch.randn(left_data.shape[0], left_data.shape[1])
        dummy_right_input = torch.randn(right_data.shape[0], right_data.shape[1])
        torch.onnx.export(model, (dummy_left_input, dummy_right_input), "join_model.onnx",
                          input_names=["left_input", "right_input"], output_names=["output"], verbose=False)

        ort_session = ort.InferenceSession("join_model.onnx")
        ort_inputs = {"left_input": left_data, "right_input": right_data}
        ort_outputs = ort_session.run(None, ort_inputs)

        result = ort_outputs[0]
        return result

# 示例数据
left_data = [
    {'id': 1, 'gender': 0},
    {'id': 2, 'gender': 1},
    {'id': 3, 'gender': 1}
]
right_data = [
    {'id': 1, 'age': 25},
    {'id': 2, 'age': 28},
    {'id': 4, 'age': 30}
]
left_table = pa.Table.from_pylist(left_data)
right_table = pa.Table.from_pylist(right_data)

# 构建查询计划
left_scan = Scan(left_table)
right_scan = Scan(right_table)
join = Join(left_scan, right_scan, 0, 0)

# 执行查询计划
while True:
    tuple = join.next()
    if tuple is None:
        break
    print(tuple)



# # 示例数据
# data = [
#     {'id': 1, 'gender': 0, 'age': 38, 'length': 177.2},
#     {'id': 2, 'gender': 1, 'age': 40, 'length': 178.8},
#     {'id': 3, 'gender': 0, 'age': 35, 'length': 175.5}
# ]
# data0 = pa.Table.from_pylist(data)
# # 构建查询计划
# scan0 = Scan(data0)
# # print(scan0.next())
# projection0 = Projection(scan0, [0, 2, 3])
# selection0 = Selection(projection0, lambda t: t[:, 1] < 39)
# # selection0 = Selection(projection0, lambda t: t[:, 2] == 177.2)

# # 执行查询计划
# while True:
#     tuple = selection0.next()
#     if tuple is None:
#         break
#     print(tuple)

