from process import *
from model import *
import torch

model = Combine_Model_SMLP(12966, 2)
model.load_state_dict(torch.load('../Results/model/net_save.pth'))

load_feature_importance(model, 'presingle_gene_')
load_feature_importance(model, 'presingle_cnn_')
load_feature_importance(model, 'presingle_all_')