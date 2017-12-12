import os
import argparse
from modules import io
from modules import vascular_data as sv
import scipy
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('global_config_file')
parser.add_argument('case_config_file')

args = parser.parse_args()

global_config_file = os.path.abspath(args.global_config_file)
case_config_file = os.path.abspath(args.case_config_file)


global_config = io.load_yaml(global_config_file)
case_config   = io.load_yaml(case_config_file)

####################################
# Get necessary params
####################################
cases = os.listdir(global_config['CASES_DIR'])
cases = [global_config['CASES_DIR']+'/'+f for f in cases if 'case.' in f]

spacing_vec = [case_config['SPACING']]*2
dims_vec    = [case_config['DIMS']]*2
ext_vec     = [case_config['DIMS']-1]*2
path_start  = case_config['PATH_START']

if case_config['LOFT']: files = open(case_config['DATA_DIR']+'/files.txt','w')
else: files = open(case_config['DATA_DIR']+'/files.txt','w')

for i, case_fn in enumerate(cases):
    case_dict = io.load_yaml(case_fn)
    print case_dict['NAME']

    image_dir = case_config['DATA_DIR']+'/'+case_dict['NAME']
    sv.mkdir(image_dir)

    image        = sv.read_mha(case_dict['IMAGE'])
    image        = sv.resample_image(image,case_config['SPACING'])

    segmentation = sv.read_mha(case_dict['SEGMENTATION'])
    segmentation = sv.resample_image(segmentation,case_config['SPACING'])

    path_dict    = sv.parsePathFile(case_dict['PATHS'])
    group_dir    = case_dict['GROUPS']

    im_np  = sv.vtk_image_to_numpy(image)
    seg_np = sv.vtk_image_to_numpy(segmentation)
    blood_np = im_np[seg_np>0.1]

    stats = {"MEAN":np.mean(im_np), "STD":np.std(im_np), "MAX":np.amax(im_np),
    "MIN":np.amin(im_np),
    "BLOOD_MEAN":np.mean(blood_np),
    "BLOOD_STD":np.std(blood_np),
    "BLOOD_MAX":np.amax(blood_np),
    "BLOOD_MIN":np.amin(blood_np)}

    for grp_id in path_dict.keys():
        path_info      = path_dict[grp_id]
        path_points    = path_info['points']
        group_name     = path_info['name']
        group_filename = group_dir +'/'+group_name

        if not os.path.exists(group_filename): continue

        group_dict = sv.parseGroupFile(group_filename)

        group_points = sorted(group_dict.keys())

        if len(group_points) < 4: continue

        tup = sv.get_segs(path_points,group_dict,
            [case_config['DIMS']]*2, [case_config['SPACING']]*2,
            case_config['NUM_CONTOUR_POINTS'])

        if tup == None: continue

        group_data_dir = image_dir+'/'+group_name
        sv.mkdir(group_data_dir)

        segs,norm_grps,interp_grps,means = tup

        im_slices  = []
        seg_slices = []

        if not case_config['LOFT']:
            for i,I in enumerate(group_points[path_start:-path_start]):
                j = i+path_start

                v = path_points[I]
                im_slice = sv.getImageReslice(image, ext_vec,
                    v[:3],v[3:6],v[6:9],True)
                seg_slice = sv.getImageReslice(segmentation, ext_vec,
                    v[:3],v[3:6],v[6:9],True)

                try:
                    np.save('{}/{}.X.npy'.format(group_data_dir,I),im_slice)
                    np.save('{}/{}.Y.npy'.format(group_data_dir,I),seg_slice)
                    np.save('{}/{}.Yc.npy'.format(group_data_dir,I),segs[j])
                    np.save('{}/{}.C.npy'.format(group_data_dir,I),norm_grps[j])
                    np.save('{}/{}.C_interp.npy'.format(group_data_dir,I),interp_grps[j])

                    scipy.misc.imsave('{}/{}.X.png'.format(group_data_dir,I),im_slice)
                    scipy.misc.imsave('{}/{}.Y.png'.format(group_data_dir,I),seg_slice)
                    scipy.misc.imsave('{}/{}.Yc.png'.format(group_data_dir,I),segs[j])

                    files.write('{}/{}\n'.format(group_data_dir,I))
                except:
                    print "failed to save {}/{}".format(group_data_dir,I)
        else:

            image_dir = case_config['DATA_DIR']+'/'+case_dict['NAME']+'_loft'
            sv.mkdir(image_dir)
            group_data_dir = image_dir+'/'+group_name
            sv.mkdir(group_data_dir)

            lofted_segs, lofted_groups = sv.loft_path_segs(interp_grps,means,
                group_dict, dims_vec,spacing_vec)

            for i in range(group_points[path_start],group_points[-path_start]):
                j = i

                v = path_points[i]
                im_slice = sv.getImageReslice(image, ext_vec,
                    v[:3],v[3:6],v[6:9],True)
                seg_slice = sv.getImageReslice(segmentation, ext_vec,
                    v[:3],v[3:6],v[6:9],True)

                try:
                    np.save('{}/{}.X.npy'.format(group_data_dir,i),im_slice)
                    np.save('{}/{}.Y.npy'.format(group_data_dir,i),seg_slice)
                    np.save('{}/{}.Yc.npy'.format(group_data_dir,i),lofted_segs[i])
                    np.save('{}/{}.C.npy'.format(group_data_dir,i),lofted_groups[i])

                    scipy.misc.imsave('{}/{}.X.png'.format(group_data_dir,i),im_slice)
                    scipy.misc.imsave('{}/{}.Y.png'.format(group_data_dir,i),seg_slice)
                    scipy.misc.imsave('{}/{}.Yc.png'.format(group_data_dir,i),lofted_segs[i])

                    files.write('{}/{}\n'.format(group_data_dir,i))
                except:
                    print "failed to save {}/{}".format(group_data_dir,i)

        io.write_csv(image_dir+'/'+'image_stats.csv',stats)

files.close()
