### 仓库文件

model_original.py为算子中不含有tensor操作的迭代器模型实现。

model_tensor.py则是算子均调用tensor操作实现，目前实现了scan、projection和selection等算子基本功能，可以执行简单物理查询计划例子。

test.ipynb在测试和调试过程中使用。

其余为测试过程中算子生成/需要调用的tensor操作模型。