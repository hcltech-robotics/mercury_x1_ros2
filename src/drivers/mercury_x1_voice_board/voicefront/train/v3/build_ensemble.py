"""Average the outputs of several openWakeWord heads (same input [N,16,96]) into one ONNX graph.
usage: python build_ensemble.py OUT.onnx a.onnx b.onnx ..."""
import sys, onnx
from onnx import helper
out, paths = sys.argv[1], sys.argv[2:]
models = [onnx.load(p) for p in paths]
base = models[0]
in_name = base.graph.input[0].name
nodes, inits, outs = [], [], []
for k, m in enumerate(models):
    g = m.graph; pre = f"m{k}_"
    ren = {i.name: pre + i.name for i in g.initializer}
    ren.update({o: pre + o for n in g.node for o in n.output})
    for i in g.initializer:
        i2 = onnx.TensorProto(); i2.CopyFrom(i); i2.name = ren[i.name]; inits.append(i2)
    for n in g.node:
        n2 = onnx.NodeProto(); n2.CopyFrom(n); n2.name = pre + (n.name or n.op_type)
        n2.ClearField("input"); n2.input.extend([ren.get(x, x) if x != g.input[0].name else in_name for x in n.input])
        n2.ClearField("output"); n2.output.extend([ren[x] for x in n.output]); nodes.append(n2)
    outs.append(ren[g.output[0].name])
nodes.append(helper.make_node("Mean", outs, [base.graph.output[0].name], name="ensemble_mean"))
graph = helper.make_graph(nodes, "hey_mercury_ensemble", [base.graph.input[0]], [base.graph.output[0]], initializer=inits)
model = helper.make_model(graph, opset_imports=base.opset_import, producer_name="build_ensemble")
model.ir_version = base.ir_version
onnx.checker.check_model(model); onnx.save(model, out); print("saved", out, "heads", len(models))
