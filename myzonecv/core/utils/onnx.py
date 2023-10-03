import onnx


def remove_initializer_inputs(onnx_file):
    """ remove inputs that are initializers """
    model = onnx.load(onnx_file)

    if model.ir_version >= 4:
        name2input = {}
        for node in model.graph.input:
            name2input[node.name] = node

        for node in model.graph.initializer:
            if node.name in name2input:
                model.graph.input.remove(name2input[node.name])

    onnx.save(model, onnx_file)


def check_noninitializer_inputs(onnx_file):
    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    input = [node.name for node in onnx_model.graph.input]
    initializer = [node.name for node in onnx_model.graph.initializer]
    feed_input = list(set(input) - set(initializer))
    assert len(feed_input) == 1
    return feed_input[0]


def print_onnx_model(onnx_file, onnx_txt=None):
    model = onnx.load(onnx_file)
    if onnx_txt is None:
        print('The ONNX model is :\n{}'.format(model))
    else:
        with open(onnx_txt, 'w') as fout:
            fout.write('The ONNX model is :\n{}'.format(model))
