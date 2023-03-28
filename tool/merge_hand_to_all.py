import torch

model_all = torch.load('snapshot_6_all.pth.tar') # module.hand_roi_net.backbone.
model_hand = torch.load('snapshot_12_hand.pth.tar') # module.hand_backbone.

dump_keys = []
for k,v in model_hand['network'].items():
    if 'module.hand_backbone' in k:
        _k = k.split('module.hand_backbone.')[1]
        save_k = 'module.hand_roi_net.backbone.' + _k
        model_all['network'][save_k] = v.cpu()
    
dump_keys = []
for k in model_all['network'].keys():
    if 'hand_position_net' in k or 'hand_rotation_net' in k:
        dump_keys.append(k)
for k in dump_keys:
    model_all['network'].pop(k)

model_all['epoch'] = 0
torch.save(model_all, 'snapshot_0.pth.tar')
