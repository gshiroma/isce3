#!/usr/bin/env python3
import time

import journal
import tempfile
from osgeo import gdal
import numpy as np
import plant

from nisar.workflows import (bandpass_insar, crossmul, dense_offsets, geo2rdr,
                             geocode_insar, h5_prep, filter_interferogram,
                             offsets_product, prepare_insar_hdf5, rdr2geo,
                             resample_slc, rubbersheet,
                             split_spectrum, unwrap, ionosphere, baseline,
                             troposphere, solid_earth_tides)

from nisar.workflows.geocode_insar import InputProduct
from nisar.workflows.insar_runconfig import InsarRunConfig
from nisar.workflows.persistence import Persistence
from nisar.workflows.yaml_argparse import YamlArgparse


def _generate_dem_with_offset(dem_file, dem_offset, scratch_dir):
    '''
    Generate DEM with offset. It may require:

    >> export GDAL_VRT_ENABLE_PYTHON=YES
    '''

    dem_with_offset = tempfile.NamedTemporaryFile(
        dir=scratch_dir, suffix='.vrt').name

    print('building DEM VRT with height offset [m]:', dem_offset)
    gdal.BuildVRT(dem_with_offset, dem_file)

    pixel_function_str = (
        '        <PixelFunctionType>offset</PixelFunctionType>\n'
        '        <PixelFunctionLanguage>Python</PixelFunctionLanguage>\n'
        f'        <PixelFunctionArguments factor="{dem_offset}"/>\n'
        '        <PixelFunctionCode><![CDATA[\n'
        'import numpy as np\n'
        'def offset(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,\n'
        ' raster_ysize, buf_radius, gt, **kwargs):\n'
        '    factor = float(kwargs["factor"])\n'
        '    out_ar[:] = in_ar[0] + factor]]>\n'
        '        </PixelFunctionCode>\n')

    with open(dem_with_offset, "r") as f:
        lines = f.readlines()

    with open(dem_with_offset, "w") as f:
        for line in lines:
            if '<VRTRasterBand' in line:
                line = line.replace('>', ' subClass="VRTDerivedRasterBand">')
            if '<SimpleSource>' in line or '<ComplexSource>' in line:
                f.write(pixel_function_str)
            f.write(line)

    return dem_with_offset


def run(cfg: dict, out_paths: dict, run_steps: dict):
    '''
    Run INSAR workflow with parameters in cfg dictionary
    '''
    info_channel = journal.info("insar.run")
    info_channel.log("starting INSAR")

    t_all = time.time()

    # if run_steps['bandpass_insar']:
    #     bandpass_insar.run(cfg)
    scratch_dir = cfg['product_path_group']['scratch_path']

    # save original parameters
    product_type_orig = cfg['primary_executable']['product_type']
    print('product_type_orig:', product_type_orig)
    # scratch_dir
    # rdr2geo, geo2rdr, coarse_resample enabled

    # k_z is computed as the InSAR phase difference using dem + 0.5 and
    # dem - 0.5

    dem_file = cfg['dynamic_ancillary_file_group']['dem_file']

    for dem_offset in [0.5, 0, -0.5]:

        if dem_offset == 0.5:
            dem_offset_str = '_p0.5'
        elif dem_offset == -0.5:
            dem_offset_str = '_m0.5'
        elif dem_offset == 0:
            dem_offset_str = '_zero'
        else:
            raise NotImplementedError("Invalid option")

        # udpate dem with dem_offset, create VRT and update dem_file ?

        # write new parameters
        cfg['primary_executable']['product_type'] = 'RIFG'

        dem_with_offset = _generate_dem_with_offset(dem_file, dem_offset,
                                                    scratch_dir)

        cfg['dynamic_ancillary_file_group']['dem_file'] = dem_with_offset

        # create RFIG writer
        if run_steps['prepare_insar_hdf5']:
            prepare_insar_hdf5.run(cfg)

        if run_steps['rdr2geo']:
            rdr2geo.run(cfg)

        if run_steps['geo2rdr']:
            geo2rdr.run(cfg)

        if run_steps['coarse_resample']:
            resample_files_dict = resample_slc.run(cfg, 'coarse', flatten=True,
                                                   flag_constant_value=True,
                                                   suffix=dem_offset_str)

            if dem_offset == 0.5:
                resample_files_dict_p_05 = resample_files_dict
            elif dem_offset == -0.5:
                resample_files_dict_m_05 = resample_files_dict

    for (_, pol_dict_p_05), (_, pol_dict_m_05) in \
            zip(resample_files_dict_p_05.items(),
                resample_files_dict_m_05.items()):

        for (_, f_p_05), (_, f_m_05) \
                in zip(pol_dict_p_05.items(), pol_dict_m_05.items()):

            print(f'DEM +0.5 file: {f_p_05}')
            print(f'DEM -0.5 file: {f_m_05}')
            data_p_05_band = gdal.Open(f_p_05, gdal.GA_ReadOnly)
            data_p_05 = data_p_05_band.GetRasterBand(1).ReadAsArray()
            data_m_05_band = gdal.Open(f_m_05, gdal.GA_ReadOnly)
            data_m_05 = data_m_05_band.GetRasterBand(1).ReadAsArray()
            data_diff = np.angle(data_p_05 * np.conj(data_m_05))
            output_filename = f_p_05.replace('_p0.5', '_kz')
            plant.save_image(data_diff, output_filename, force=True)
            print(f'file saved: {output_filename}')

        print('*** resample_files_dict:', resample_files_dict)

    # if run_steps['baseline']:
    #     baseline.run(cfg, out_paths)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran INSAR in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    # parse CLI input
    yaml_parser = YamlArgparse()
    args = yaml_parser.parse()

    # convert CLI input to run configuration
    insar_runcfg = InsarRunConfig(args)

    # To allow persistence, a logfile is required. Raise exception
    # if logfile is None and persistence is requested
    logfile_path = insar_runcfg.cfg['logging']['path']
    if (logfile_path is None) and insar_runcfg.args.restart:
        raise ValueError('InSAR workflow persistence requires to specify a logfile')
    persist = Persistence(logfile_path, insar_runcfg.args.restart)

    # run InSAR workflow
    if persist.run:
        _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

        run(insar_runcfg.cfg, out_paths, persist.run_steps)
