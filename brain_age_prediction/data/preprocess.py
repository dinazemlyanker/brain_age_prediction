import os
import argparse
import numpy as np
# import voxelmorph as vxm
import torch
# import surfa as sf
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, distance_transform_edt, binary_fill_holes
from scipy.ndimage import label as scipy_label
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
import subprocess


def getM(ref, mov):

    zmat = np.zeros(ref.shape[::-1])
    zcol = np.zeros([ref.shape[1], 1])
    ocol = np.ones([ref.shape[1], 1])
    zero = np.zeros(zmat.shape)

    A = np.concatenate([
        np.concatenate([np.transpose(ref), zero, zero, ocol, zcol, zcol], axis=1),
        np.concatenate([zero, np.transpose(ref), zero, zcol, ocol, zcol], axis=1),
        np.concatenate([zero, zero, np.transpose(ref), zcol, zcol, ocol], axis=1)], axis=0)

    b = np.concatenate([np.transpose(mov[0, :]), np.transpose(mov[1, :]), np.transpose(mov[2, :])], axis=0)

    x = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A), b))

    M = np.stack([
        [x[0], x[1], x[2], x[9]],
        [x[3], x[4], x[5], x[10]],
        [x[6], x[7], x[8], x[11]],
        [0, 0, 0, 1]])

    return M

def load_volume(path_volume, im_only=True, squeeze=True, dtype=None, aff_ref=None, to_skull_strip=None):

    assert path_volume.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file: %s' % path_volume

    if path_volume.endswith(('.nii', '.nii.gz', '.mgz')):
        x = nib.load(path_volume)
        if to_skull_strip is not None:
            seg_data = nib.load(to_skull_strip)
            brain_mask = (seg_data > 0)
            x = x * brain_mask
        if squeeze:
            volume = np.squeeze(x.get_fdata())
        else:
            volume = x.get_fdata()
        aff = x.affine
        header = x.header
    else:  # npz
        volume = np.load(path_volume)['vol_data']
        if squeeze:
            volume = np.squeeze(volume)
        aff = np.eye(4)
        header = nib.Nifti1Header()
    if dtype is not None:
        if 'int' in dtype:
            volume = np.round(volume)
        volume = volume.astype(dtype=dtype)

    # align image to reference affine matrix
    if aff_ref is not None:
        n_dims, _ = get_dims(list(volume.shape), max_channels=10)
        volume, aff = align_volume_to_ref(volume, aff, aff_ref=aff_ref, return_aff=True, n_dims=n_dims)

    if im_only:
        return volume
    else:
        return volume, aff, header

def fast_3D_interp_torch(X, II, JJ, KK, mode):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = X[IIr, JJr, KKr]
    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        Y = torch.zeros(II.shape, device='cpu')
        Y[ok] = c.float()

    else:
        sf.system.fatal('mode must be linear or nearest')

    return Y

def save_volume(volume, aff, header, path, res=None, dtype=None, n_dims=3):
    # os.mkdir(os.path.dirname(path))
    if '.npz' in path:
        np.savez_compressed(path, vol_data=volume)
    else:
        if header is None:
            header = nib.Nifti1Header()
        if isinstance(aff, str):
            if aff == 'FS':
                aff = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        elif aff is None:
            aff = np.eye(4)
        nifty = nib.Nifti1Image(volume, aff, header)
        if dtype is not None:
            if 'int' in dtype:
                volume = np.round(volume)
            volume = volume.astype(dtype=dtype)
            nifty.set_data_dtype(dtype)
        if res is not None:
            if n_dims is None:
                n_dims, _ = get_dims(volume.shape)
            res = reformat_to_list(res, length=n_dims, dtype=None)
            nifty.header.set_zooms(res)
        nib.save(nifty, path)





def easy_reg(synseg_file, synthsr_file, output_file, to_skull_strip=None):
    # path labels
    atlas_volsize = [160, 160, 192]
    atlas_aff = np.matrix([[-1, 0, 0, 79], [0, 0, 1, -104], [0, -1, 0, 79], [0, 0, 0, 1]])
    labels = np.array([2,4,5,7,8,10,11,12,13,14,15,16,17,18,26,28,41,43,44,46,47,49,50,51,52,53,54,58,60,
                                        1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,
                                        2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035])
    nlab = len(labels)
    atlasCOG = np.array([[-28.,-18.,-37.,-19.,-27.,-19.,-23.,-31.,-26.,-2.,-3.,-3.,-29.,-26.,-14.,-14.,24.,14.,31.,12.,18.,14.,19.,26.,21.,25.,22.,11.,8.,-52.,-6.,-36.,-7.,-24.,-37.,-39.,-52.,-9.,-27.,-26.,-14.,-8.,-59.,-28.,-7.,-49.,-43.,-47.,-12.,-46.,-6.,-43.,-10.,-7.,-33.,-11.,-23.,-55.,-50.,-10.,-29.,-46.,-38.,48.,4.,31.,3.,21.,33.,37.,47.,3.,24.,20.,8.,4.,54.,21.,5.,45.,38.,46.,8.,45.,3.,38.,6.,4.,29.,9.,19.,51.,49.,10.,24.,43.,33.],
                        [-30.,-17.,-13.,-36.,-40.,-22.,-3.,-5.,-9.,-14.,-31.,-21.,-15.,-1.,3.,-16.,-32.,-20.,-14.,-37.,-42.,-24.,-3.,-6.,-10.,-15.,-2.,3.,-17.,-44.,-5.,-15.,-71.,2.,-29.,-70.,-23.,-44.,-73.,22.,-57.,27.,-19.,-23.,-45.,4.,31.,20.,-68.,-38.,-33.,-26.,-60.,23.,22.,0.,-72.,-12.,-49.,49.,17.,-25.,-3.,-42.,-1.,-16.,-76.,0.,-34.,-69.,-16.,-44.,-73.,22.,-56.,28.,-18.,-25.,-45.,-3.,30.,14.,-69.,-37.,-32.,-30.,-60.,21.,21.,0.,-72.,-11.,-49.,48.,15.,-27.,-3.],
                        [12.,14.,-13.,-41.,-51.,1.,13.,3.,1.,0.,-40.,-28.,-15.,-10.,2.,-7.,11.,14.,-12.,-40.,-51.,2.,14.,4.,2.,-14.,-10.,4.,-7.,-8.,32.,40.,-14.,-21.,-28.,-4.,-28.,-3.,-35.,3.,-29.,4.,-17.,-21.,35.,18.,9.,20.,-24.,28.,25.,34.,7.,18.,35.,48.,16.,-5.,12.,22.,-18.,1.,4.,-12.,32.,43.,-11.,-21.,-29.,-3.,-27.,0.,-34.,3.,-25.,6.,-18.,-20.,36.,18.,11.,20.,-20.,26.,25.,34.,4.,24.,34.,47.,17.,-5.,10.,20.,-18.,0.,4.]])

    synseg_buffer, synseg_aff, synseg_h = load_volume(synseg_file, im_only=False, squeeze=True, dtype=None, aff_ref=None, to_skull_strip=None)

    synsegCOG = np.zeros([4, nlab])
    ok = np.ones(nlab)
    for l in range(nlab):
        aux = np.where(synseg_buffer == labels[l])
        if len(aux[0]) > 50:
            synsegCOG[0, l] = np.median(aux[0])
            synsegCOG[1, l] = np.median(aux[1])
            synsegCOG[2, l] = np.median(aux[2])
            synsegCOG[3, l] = 1
        else:
            ok[l] = 0
    synsegCOG = np.matmul(synseg_aff, synsegCOG)[:-1, :]
    Mref = getM(atlasCOG[:, ok > 0], synsegCOG[:, ok > 0])

    II, JJ, KK = np.meshgrid(np.arange(atlas_volsize[0]), np.arange(atlas_volsize[1]), np.arange(atlas_volsize[2]), indexing='ij')
    II = torch.tensor(II, device='cpu')
    JJ = torch.tensor(JJ, device='cpu')
    KK = torch.tensor(KK, device='cpu')
    synsr, synsr_aff, synsr_h = load_volume(synthsr_file, im_only=False, squeeze=True, dtype=None, aff_ref=None, to_skull_strip=to_skull_strip)
    synsr = torch.tensor(synsr)
    affine = torch.tensor(np.matmul(np.linalg.inv(synsr_aff), np.matmul(Mref, atlas_aff)), device='cpu')
    II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
    JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
    KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
    reg_synthsr = fast_3D_interp_torch(synsr, II2, JJ2, KK2, 'linear')

    save_volume(reg_synthsr, atlas_aff, synsr_h, output_file)    
    

def preprocess(synseg_file, synthsr_file, output_file, to_skull_strip=None):
    easy_reg(synseg_file, synthsr_file, output_file)

def run_synthsr(file_df):
    print('Running SynthSR..........................................................')
    synthsr_dir = 'synthsr_output'
    synthsr_output_paths = []
    if not os.path.exists(synthsr_dir):
        os.mkdir(synthsr_dir)
    for _,row in file_df.iterrows():
        filename = row['file_name']
        filepath = row['filepath']
        synthsr_output_fn = os.path.join(synthsr_dir, filename.replace('.nii.gz', '_synthsr.nii.gz'))
        synthsr_output_paths.append(synthsr_output_fn)
        if not os.path.exists(synthsr_output_fn):
            subprocess.run(['mri_synthsr', '--i', filepath, '--o', synthsr_output_fn, "--threads", '4'])
    return synthsr_output_paths

def run_synthseg(file_df):
    print('Running SynthSeg.........................................................')
    synthseg_dir = 'synthseg_output'
    synthseg_output_paths = []
    if not os.path.exists(synthseg_dir):
        os.mkdir(synthseg_dir)
    for _,row in file_df.iterrows():
        filename = row['file_name']
        filepath = row['synthsr_path']
        synthseg_output_fn = os.path.join(synthseg_dir, filename +  '_synthseg.nii.gz')
        synthseg_output_paths.append(synthseg_output_fn)
        if not os.path.exists(synthseg_output_fn):
            subprocess.run(['mri_synthseg', '--i', filepath, '--o', synthseg_output_fn, '--threads', '4'])
            print(['mri_synthseg', '--i', filepath, '--o', synthseg_output_fn, '--threads', '4'])
    return synthseg_output_paths


def preprocess_data(input_file_csv_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df = pd.read_csv(input_file_csv_path)
    output_file_names = []
    # TODO: fix this so that it makes sense
    df['file_name'] = df['synthsr_path'].apply(lambda fp: '_'.join(fp.split('/')[-3:]))
    df['file_name'] = df['file_name'].apply(lambda fp: fp.replace('.nii.gz', ''))
    print(df)
    if 'synthsr_path' not in df.columns:
        synthsr_output_paths = run_synthsr(df)
        df['synthsr_path'] = synthsr_output_paths
    if 'synthseg_path' not in df.columns:
        synthseg_output_paths = run_synthseg(df)
        df['synthseg_path'] = synthseg_output_paths
    
    output_file_names = []
    for _, row in df.iterrows():
        filepath = row['synthsr_path']
        synthseg = row['synthseg_path']
        output_file = os.path.join(output_dir, row['file_name'] + '.nii.gz')
        output_file_names.append(output_file)
        if os.path.exists(synthseg) and os.path.exists(filepath) and not os.path.exists(output_file):
            print('Preprocessing: ', output_file)
            try:
                easy_reg(synthseg, filepath, output_file)
            except:
                print('ERRORED on: ', filepath)
    df['preprocessed_path'] = output_file_names
    df.to_csv(input_file_csv_path)


    









    




