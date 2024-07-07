class Operator:
    def __init__(self):
        self.child = None

    def next(self):
        pass

class Scan(Operator):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.index = 0

    def next(self):
        if self.index < len(self.data):
            result = self.data[self.index]
            self.index += 1
            return result
        else:
            return None

class Join(Operator):
    def __init__(self, left_child, right_child, join_condition):
        super().__init__()
        self.left_child = left_child
        self.right_child = right_child
        self.join_condition = join_condition
        self.left_tuple = None
        self.right_tuple = None

    def next(self):
        while True:
            if self.left_tuple is None:
                self.left_tuple = self.left_child.next()
                if self.left_tuple is None:
                    return None
                self.right_child.index = 0

            self.right_tuple = self.right_child.next()
            if self.right_tuple is None:
                self.left_tuple = None
                continue

            if self.join_condition(self.left_tuple, self.right_tuple):
                result = {**self.left_tuple, **self.right_tuple}
                return result

class Projection(Operator):
    def __init__(self, child, columns):
        super().__init__()
        self.child = child
        self.columns = columns

    def next(self):
        tuple = self.child.next()
        if tuple is None:
            return None
        
        result = {column: tuple[column] for column in self.columns}
        return result

class Selection(Operator):
    def __init__(self, child, predicate):
        super().__init__()
        self.child = child
        self.predicate = predicate

    def next(self):
        while True:
            tuple = self.child.next()
            if tuple is None:
                return None
            
            if self.predicate(tuple):
                return tuple

# 示例数据
data1 = [
    {'id': 1, 'name': 'John', 'age': 38},
    {'id': 2, 'name': 'Jane', 'age': 40},
    {'id': 3, 'name': 'Bob', 'age': 35}
]

data2 = [
    {'id': 1, 'city': 'New York'},
    {'id': 2, 'city': 'London'},
    {'id': 3, 'city': 'Paris'}
]

# 构建查询计划
scan1 = Scan(data1)
scan2 = Scan(data2)
join = Join(scan1, scan2, lambda t1, t2: t1['id'] == t2['id'])
projection = Projection(join, ['name', 'age', 'city'])
selection = Selection(projection, lambda t: t['age'] <= 39)

# 执行查询计划
while True:
    tuple = selection.next()
    if tuple is None:
        break
    print(tuple)
# scan0 = Scan(data1)
# projection0 = Projection(scan0, ['name','age'])
# # selection0 = Selection(projection0, lambda t: t['age'] < 38)
# selection0 = Selection(projection0, lambda t: t['age'] < 38 & t['age'] > 35)
# while True:
#     tuple = selection0.next()
#     if tuple is None:
#         break
#     print(tuple)