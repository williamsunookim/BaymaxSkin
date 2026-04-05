import torch
from verify_series_model import jointPointcloud

model = jointPointcloud()
model.load_state_dict(torch.load('series_model_best.pt', map_location=torch.device('cpu')))
model.eval()
dummy = torch.zeros(1, 16)
torch.onnx.export(model, dummy, 'static/models/series_model_best.onnx',
                  input_names=['input'], output_names=['output'])
