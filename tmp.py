import onnx
from onnxruntime.quantization import QuantType
from onnxruntime.quantization import quantize_dynamic

model_fp32 = 'static/models/series_model_best.onnx'
model_quant = 'static/models/series_model_best_quant.onnx'

# 동적 양자화 수행
quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)