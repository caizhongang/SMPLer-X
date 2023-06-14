import csv
import pandas as pd
import os.path as osp

thr = 90
col = 2
content = ['index', 'img_path', 'pve', 'pa_pve']

home = '/mnt/lustrenew/share_data/zoetrope/osx/output_wanqi'

# EHF
# h4w_path = 'test_h4w_vis_ep7_EHF_20230606_180847/vis'
# osx_path = 'test_osx_vis_ep999_EHF_20230606_174828/vis'
# our_l32_path = 'test_exp114_vis_ep3_EHF_20230607_164910/vis'
# our_h20_path = 'test_exp117_vis_ep4_EHF_20230607_164825/vis'
# csv_name = 'ehf_smplx_error.csv'

# Agora
# osx_path = 'test_osx_vis_ep999_AGORA_val_20230607_144123/vis'
# h4w_path = 'test_h4w_vis_ep7_AGORA_val_20230607_145313/vis'
# our_h20_path = 'test_exp117_vis_ep4_AGORA_val_20230607_160510/vis'
# our_l32_path = 'test_exp114_vis_ep3_AGORA_val_20230607_151156/vis'
# csv_name = 'agora_smplx_error.csv'

# # UBody
# osx_path = 'test_osx_vis_ep999_UBody_20230607_145041/vis'
# h4w_path = 'test_h4w_vis_ep7_UBody_20230607_150602/vis'
# our_h20_path = 'test_exp117_vis_ep4_UBody_20230607_153854/vis'
# our_l32_path = 'test_exp114_vis_ep3_UBody_20230607_153803/vis'
# csv_name = 'UBody_smplx_error.csv'

# # Ego
# h4w_path = 'test_h4w_vis_ep7_EgoBody_Egocentric_20230607_153925/vis'
# osx_path = 'test_osx_vis_ep999_EgoBody_Egocentric_20230607_152442/vis'
# our_l32_path = 'test_exp114_vis_ep3_EgoBody_Egocentric_20230607_163538/vis'
# our_h20_path = 'test_exp117_vis_ep4_EgoBody_Egocentric_20230607_163546/vis'
# csv_name = 'EgoBody_Egocentric_smplx_error.csv'

# # ARCTIC
# h4w_path = 'test_h4w_vis_ep7_ARCTIC_20230607_184338/vis'
# osx_path = 'test_osx_vis_ep999_ARCTIC_20230607_182625/vis'
# our_l32_path = 'test_exp114_vis_ep3_ARCTIC_20230607_203528/vis'
# our_h20_path = 'test_exp117_vis_ep4_ARCTIC_20230607_204153/vis'
# csv_name = 'ARCTIC_smplx_error.csv'

# RenBody
h4w_path = 'test_h4w_vis_ep7_RenBody_20230607_213652/vis'
osx_path = 'test_osx_vis_ep999_RenBody_20230607_200129/vis'
our_l32_path = 'test_exp114_vis_ep3_RenBody_20230607_214327/vis'
our_h20_path = 'test_exp117_vis_ep4_RenBody_20230607_212645/vis'
csv_name = 'RenBody_HiRes_smplx_error.csv'

select_csv_file = f'./select_{csv_name}'
file = open(select_csv_file, 'a', newline='')
writer = csv.writer(file)
header = [f'index_[thr: {thr}]', 'img_path', f'H4W_{content[col]}', f'OSX_{content[col]}', 
    f'l32_{content[col]}', f'H20_{content[col]}']
writer.writerow(header)

h4w_file = osp.join(home, h4w_path, csv_name)
osx_file = osp.join(home, osx_path, csv_name)
our_l32_file = osp.join(home, our_l32_path, csv_name)
our_h20_file = osp.join(home, our_h20_path, csv_name)
# import pdb;pdb.set_trace()

h4w_file_data = pd.read_csv(h4w_file)
h4w_error = h4w_file_data.iloc[:, col]

osx_file_data = pd.read_csv(osx_file)
osx_error = osx_file_data.iloc[:, col]

our_l32_file_data = pd.read_csv(our_l32_file)
our_l32_error = our_l32_file_data.iloc[:, col]

our_h20_file_data = pd.read_csv(our_h20_file)
our_h20_error = our_h20_file_data.iloc[:, col]
# import pdb; pdb.set_trace()
for i in range(len(h4w_error)):
    delta1 = h4w_error[i] - our_h20_error[i]
    delta2 = osx_error[i] - our_h20_error[i]
    delta3 = our_h20_error[i] - our_l32_error[i]

    if delta1 > thr and delta2 > thr and abs(delta3) < 5 and h4w_error[i] < 200 and osx_error[i] < 200:
        print(our_h20_file_data.iloc[i, 1], delta1, delta2, delta3)
        new_line = [our_h20_file_data.iloc[i, 0], our_h20_file_data.iloc[i, 1], 
            h4w_error[i], osx_error[i], our_l32_error[i], our_h20_error[i]]
        # Append the new line to the CSV file
        writer.writerow(new_line)

file.close()





