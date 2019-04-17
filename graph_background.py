import numpy as np

//Basic operations
class Operation():
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass

class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        return x_var + y_var

class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        return x_var * y_var

class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        return x_var.dot(y_var)

class Sigmoid(Operation):
    def __init__(self, z):
        super.__init__([z])

    def compute(self, z_var):
        return 1. / (1 + np.exp(-z_var))


//Basic Placeholders && Variables
class Placeholder():
    def __init__(self):
        self.output_nodes = []

        _default_graph.placeholders.append(self)

class Variable():
    def __init__(self, initial_value = None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)

//Graph && Session
class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph

        _default_graph = self

def traverse_postoder(operation):
    node_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        node_postorder.append(node)

    recurse(operation)
    return node_postorder


//Session running using traverse_postoder
class Session():
    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postoder(operation)
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs) # args
        if type(node.output) == list:
            node.output = np.array(node.output)

        return operation.output

# g = Graph()
# g.set_as_default()
# A = Variable(np.array([[10, 20], [30, 40]]))
# b = Variable(np.array([1, 2]))
# x = Placeholder()
# y = matmul(A, x)
# z = add(y, b)
#
# sess = Session()
# result = sess.run(operation = z, feed_dict = {x:10})
