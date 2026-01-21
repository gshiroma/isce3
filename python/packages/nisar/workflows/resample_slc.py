#!/usr/bin/env python3

'''
wrapper for resample
'''

import os
import pathlib
import time
import tempfile

import journal
import isce3
from osgeo import gdal
from nisar.products.readers import SLC
from nisar.workflows.helpers import copy_raster, complex_raster_path_from_h5
from nisar.workflows.resample_slc_runconfig import ResampleSlcRunConfig
from nisar.workflows.yaml_argparse import YamlArgparse


def build_empty_vrt(filename, length, width, fill_value, dtype='CFloat32'):
    """Build an empty VRT file, i.e, not pointing to any rasters,
       with given input dimensions (length and width), data type, and
       fill value.
       Parameters
       ----------
       filename: str
              VRT file name
       length: int
              VRT data length
       width: int
              VRT data width
       fill_value: scalar
              VRT data fill value
       dtype: str
              VRT data type
    """

    vrt_contents = f'<VRTDataset rasterXSize="{width}"'
    vrt_contents += f' rasterYSize="{length}"> \n'

    vrt_contents += (
        f'  <VRTRasterBand dataType="{dtype}" band="1"> \n'
        f'    <NoDataValue>{fill_value}</NoDataValue> \n'
        f'    <HideNoDataValue>{fill_value}</HideNoDataValue> \n'
        f'  </VRTRasterBand> \n'
        f'</VRTDataset> \n')

    with open(filename, 'w') as out:
        out.write(vrt_contents)

    if os.path.isfile(filename):
        print('file saved:', filename)


def run(cfg, resample_type, flatten=False, flag_constant_value=False,
        suffix=''):
    '''
    run resample_slc
    '''
    input_hdf5 = cfg['input_file_group']['secondary_rslc_file']
    scratch_path = pathlib.Path(cfg['product_path_group']['scratch_path'])
    freq_pols = cfg['processing']['input_subset']['list_of_frequencies']

    # According to the type of resampling, choose proper resample cfg
    resamp_args = cfg['processing'][f'{resample_type}_resample']

    # Get SLC parameters
    slc = SLC(hdf5file=input_hdf5)
    # If flattening is enabled, load the reference SLC
    if flatten:
        reference_hdf5 = cfg['input_file_group']['reference_rslc_file']
        ref_slc = SLC(hdf5file=reference_hdf5)

    info_channel = journal.info('resample_slc.run')
    info_channel.log('starting resampling SLC')

    # Check if use GPU or CPU resampling
    use_gpu = isce3.core.gpu_check.use_gpu(cfg['worker']['gpu_enabled'],
                                           cfg['worker']['gpu_id'])

    if use_gpu:
        # Set current CUDA device
        device = isce3.cuda.core.Device(cfg['worker']['gpu_id'])
        isce3.cuda.core.set_device(device)

    t_all = time.time()

    resample_files_dict = {}

    for freq in freq_pols.keys():

        resample_files_dict[freq] = {}

        # Get frequency specific parameters
        radar_grid = slc.getRadarGrid(frequency=freq)
        native_doppler = slc.getDopplerCentroid(frequency=freq)

        # Open offsets
        offsets_dir = pathlib.Path(resamp_args['offsets_dir'])

        # Create separate directories for coarse and fine resample
        # Open corresponding range/azimuth offsets
        resample_slc_scratch_path = scratch_path / \
                                    f'{resample_type}_resample_slc' / f'freq{freq}'
        if resample_type == 'coarse':
            offsets_path = offsets_dir / 'geo2rdr' / f'freq{freq}'
        else:
            # We checked the existence of HH/VV offsets in resample_slc_runconfig.py
            # Select the first offsets available between HH and VV
            freq_offsets_path = offsets_dir / 'rubbersheet_offsets' / f'freq{freq}'
            if os.path.isdir(str(freq_offsets_path/'HH')):
                offsets_path = freq_offsets_path/'HH'
            else:
                offsets_path = freq_offsets_path/'VV'
        rg_off = isce3.io.Raster(str(offsets_path / 'range.off'))
        az_off = isce3.io.Raster(str(offsets_path / 'azimuth.off'))

        # Create resample slc directory
        resample_slc_scratch_path.mkdir(parents=True, exist_ok=True)

        # Initialize CPU or GPU resample object accordingly
        if use_gpu:
            Resamp = isce3.cuda.image.ResampSlc
        else:
            Resamp = isce3.image.ResampSlc

        # If flattening is enabled, add the reference SLC radar grid to
        # the call to the constructor of the ResampSlc module
        if flatten:
            ref_radar_grid = ref_slc.getRadarGrid(frequency=freq)
            resamp_obj = Resamp(radar_grid, native_doppler,
                                ref_rdr_grid=ref_radar_grid)
        else:
            resamp_obj = Resamp(radar_grid, native_doppler)

        # If lines per tile is > 0, assign it to resamp_obj
        if resamp_args['lines_per_tile']:
            resamp_obj.lines_per_tile = resamp_args['lines_per_tile']

        # Get polarization list for which resample SLCs
        pol_list = freq_pols[freq]

        for pol in pol_list:
            # Create directory for each polarization
            out_dir = resample_slc_scratch_path / pol
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f'coregistered_secondary{suffix}.slc'

            # If necessary, perform complex32 to complex64 conversion on input
            input_as_c32_path = str(out_dir/'secondary.slc')
            input_raster_path, _ = complex_raster_path_from_h5(
                slc, freq, pol, input_hdf5, resamp_args['lines_per_tile'],
                input_as_c32_path)

            input_raster = isce3.io.Raster(input_raster_path)

            # Dump secondary RSLC on disk
            # raster_path = str(out_dir/'secondary.slc')
            # copy_raster(input_hdf5, freq, pol, 1000,
            #            raster_path, file_type='ENVI')
            # input_raster = isce3.io.Raster(raster_path)

            # Dump secondary RSLC on disk
            # raster_path = str(out_dir/'secondary.slc')
            # copy_raster(input_hdf5, freq, pol, 1000,
            #            raster_path, file_type='ENVI')
            # input_raster = isce3.io.Raster(raster_path)

            if flag_constant_value:
                print('*** constant value!')
                fill_value = 1
                input_raster_vrt = tempfile.NamedTemporaryFile(
                    dir=scratch_path, suffix='.vrt').name
                build_empty_vrt(input_raster_vrt, input_raster.length,
                                input_raster.width, fill_value)
                input_raster = isce3.io.Raster(input_raster_vrt)

            # Create output raster
            resamp_slc = isce3.io.Raster(str(out_path), rg_off.width,
                                         rg_off.length, rg_off.num_bands,
                                         gdal.GDT_CFloat32, 'ENVI')
            resamp_obj.resamp(input_raster, resamp_slc, rg_off, az_off,
                              flatten=flatten)

            resample_files_dict[freq][pol] = str(out_path)

    t_all_elapsed = time.time() - t_all
    info_channel.log(f"successfully ran resample in {t_all_elapsed:.3f} seconds")


if __name__ == "__main__":
    '''
    run resample_slc from command line
    '''

    # load command line args
    resample_slc_parser = YamlArgparse(resample_type=True)
    args = resample_slc_parser.parse()

    # Extract resample_type
    resample_type = args.resample_type

    # Get a runconfig dictionary from command line args
    resample_slc_runconfig = ResampleSlcRunConfig(args, resample_type)

    # Run resample_slc
    run(resample_slc_runconfig.cfg, resample_type)
