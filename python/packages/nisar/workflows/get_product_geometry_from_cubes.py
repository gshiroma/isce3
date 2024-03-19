#!/usr/bin/env python3

import os
import argparse
from osgeo import gdal
import isce3
from nisar.products.readers import open_product
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# The import below was commented to make this script standalone
# from get_product_geometry import (geometry_datasets_dict,
#                                   get_dem_interp_method)


def get_dem_interp_method(dem_interp_method):
    if (dem_interp_method is None or
            dem_interp_method == 'BIQUINTIC'):
        return isce3.core.DataInterpMethod.BIQUINTIC
    if (dem_interp_method == 'SINC'):
        return isce3.core.DataInterpMethod.SINC
    if (dem_interp_method == 'BILINEAR'):
        return isce3.core.DataInterpMethod.BILINEAR
    if (dem_interp_method == 'BICUBIC'):
        return isce3.core.DataInterpMethod.BICUBIC
    if (dem_interp_method == 'NEAREST'):
        return isce3.core.DataInterpMethod.NEAREST
    error_msg = f'ERROR invalid DEM interpolation method: {dem_interp_method}'
    raise NotImplementedError(error_msg)


geometry_datasets_dict = {
    'interpolated_dem': 'interpolatedDem',
    'slant_range': 'slantRange',
    'azimuth_time': 'zeroDopplerAzimuthTime',
    'incidence_angle': 'incidenceAngle',
    'los_unit_vector_x': 'losUnitVectorX',
    'los_unit_vector_y': 'losUnitVectorY',
    'along_track_unit_vector_x': 'alongTrackUnitVectorX',
    'along_track_unit_vector_y': 'alongTrackUnitVectorY',
    'elevation_angle': 'elevationAngle',
    'ground_track_velocity': 'groundTrackVelocity',
    'local_incidence_angle': 'localIncidenceAngle',
    'projection_angle': 'projectionAngle',
    'simulated_radar_brightness': 'simulatedRadarBrightness',
    'coordinate_x': 'coordinateX',
    'coordinate_y': 'coordinateY'
}


def get_parser():
    '''
    Command line parser.
    '''
    descr = 'Get product geometry from metadata cubes'
    epilog = ''
    parser = argparse.ArgumentParser(epilog=epilog,
                                     description=descr)

    parser.add_argument(type=str,
                        dest='input_file',
                        default=None,
                        help='Input NISAR L2 file')

    parser.add_argument('--dem',
                        '--dem-file',
                        dest='dem_file',
                        required=True,
                        type=str,
                        help='Reference DEM file')

    parser.add_argument('--od',
                        '--output-dir',
                        dest='output_dir',
                        type=str,
                        default='.',
                        help='Output directory')                        

    parser.add_argument('--frequency',
                        '--freq',
                        type=str,
                        default='A',
                        dest='frequency',
                        choices=['A', 'B'],
                        help='Frequency band: "A" or "B"')

    parser.add_argument('--dem-interp-method',
                        dest='dem_interp_method',
                        type=str,
                        choices=['SINC', 'BILINEAR', 'BICUBIC', 'NEAREST',
                                 'BIQUINTIC'],
                        help='DEM interpolation method. Options:'
                        ' "SINC", "BILINEAR", "BICUBIC", "NEAREST", and'
                        ' "BIQUINTIC"')

    parser.add_argument('--threshold-geo2rdr',
                        '--geo2rdr-threshold',
                        type=float,
                        dest='threshold_geo2rdr',
                        help='Convergence threshold for geo2rdr')

    parser.add_argument('--num-iter-geo2rdr',
                        '--geo2rdr-num-iter',
                        type=int,
                        dest='num_iter_geo2rdr',
                        help='Maximum number of iterations for geo2rdr')

    parser.add_argument('--delta-range-geo2rdr',
                        '--geo2rdr-delta-range',
                        type=float,
                        dest='delta_range_geo2rdr',
                        help='Delta range for geo2rdr')

    parser.add_argument('--out-interpolated-dem',
                        action='store_true',
                        dest='flag_interpolated_dem',
                        help='Save interpolated DEM')

    parser.add_argument('--out-slant-range',
                        action='store_true',
                        dest='flag_slant_range',
                        help='Save slant-range')

    parser.add_argument('--out-azimuth-time',
                        '--out-az-time',
                        action='store_true',
                        dest='flag_azimuth_time',
                        help='Save azimuth time')

    parser.add_argument('--out-inc-angle',
                        '--out-incidence-angle',
                        action='store_true',
                        dest='flag_incidence_angle',
                        help='Save interpolated DEM')

    parser.add_argument('--out-line-of-sight',
                        '--out-los',
                        action='store_true',
                        dest='flag_los',
                        help='Save line-of-sight unit vector')

    parser.add_argument('--out-along-track',
                        action='store_true',
                        dest='flag_along_track',
                        help='Save along-track unit vector')

    parser.add_argument('--out-elevation-angle',
                        action='store_true',
                        dest='flag_elevation_angle',
                        help='Save elevation angle')

    parser.add_argument('--out-ground-track-velocity',
                        action='store_true',
                        dest='flag_ground_track_velocity',
                        help='Save ground track velocity')

    return parser.parse_args()


def run(args):
    '''
    run main method
    '''
    # Get NISAR product
    nisar_product_obj = open_product(args.input_file)
    if nisar_product_obj.getProductLevel() == 'L2':
        interpolate_radar_grid(nisar_product_obj, args)
    else:
        raise NotImplementedError


def interpolate_radar_grid(nisar_product_obj, args):
    '''
    interpolate radar grid from L2 products' metadata cubes
    '''
    if args.frequency and args.frequency == 'B':
        frequency_str = 'B'
    else:
        frequency_str = 'A'

    orbit = nisar_product_obj.getOrbit()

    # Get GeoGridProduct obj and lookside
    try:
        geogrid_product_obj = nisar_product_obj.getGeoGridProduct()
    except AttributeError:
        error_message = ('ERROR get_product geometry_from_cues.py does not'
                         ' support product type'
                         f' "{nisar_product_obj.productType}".')
        raise NotImplementedError(error_message)

    lookside = geogrid_product_obj.lookside

    # Get Grid obj, GeoGrid obj, and wavelength
    grid_obj = nisar_product_obj.getGridMetadata(frequency_str)
    geogrid_obj = grid_obj.geogrid
    wavelength = grid_obj.wavelength

    # Get grid Doppler (zero-Doppler) and native Doppler LUTs
    grid_doppler = isce3.core.LUT2d()
    native_doppler = nisar_product_obj.getDopplerCentroid()
    native_doppler.bounds_error = False

    nbands = 1
    shape = [nbands, geogrid_obj.length, geogrid_obj.width]
    if args.output_dir and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    dem_raster = isce3.io.Raster(args.dem_file)

    output_file_list = []
    output_obj_list = []

    flag_all = (args.flag_interpolated_dem is not True and
                args.flag_slant_range is not True and
                args.flag_azimuth_time is not True and
                args.flag_incidence_angle is not True and
                args.flag_los is not True and
                args.flag_along_track is not True and
                args.flag_elevation_angle is not True and
                args.flag_ground_track_velocity is not True)

    args.flag_interpolated_dem |= flag_all

    # Interpolate DEM
    interpolated_dem_raster, interpolated_dem_file = _get_raster(
        args.output_dir, geometry_datasets_dict['interpolated_dem'],
        gdal.GDT_Float32, shape,
        output_file_list, output_obj_list)

    dem_interp_method = get_dem_interp_method(args.dem_interp_method)

    geo2rdr_params = isce3.geometry.Geo2RdrParams()

    if args.threshold_geo2rdr is not None:
        geo2rdr_params.threshold = args.threshold_geo2rdr
    if args.num_iter_geo2rdr is not None:
        geo2rdr_params.maxiter = args.num_iter_geo2rdr
    if args.delta_range_geo2rdr is not None:
        geo2rdr_params.delta_range = args.delta_range_geo2rdr

    # interpolate the DEM over the output geogrid
    isce3.geogrid.get_radar_grid(lookside,
                                 wavelength,
                                 dem_raster,
                                 geogrid_obj,
                                 orbit,
                                 native_doppler,
                                 grid_doppler,
                                 dem_interp_method,
                                 geo2rdr_params,
                                 interpolated_dem_raster)

    # Flush data
    for obj in output_obj_list:
        try:
            obj.close_dataset()
        except:
            pass
        del obj

    # Create list of cubes to process
    cube_dataset_names_list = []
    _update_cube_dataset_names_list(
        geometry_datasets_dict['slant_range'],
        args.flag_slant_range, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['azimuth_time'],
        args.flag_azimuth_time, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['incidence_angle'],
        args.flag_incidence_angle, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['los_unit_vector_x'],
        args.flag_los, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['los_unit_vector_y'],
        args.flag_los, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['along_track_unit_vector_x'],
        args.flag_along_track, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['along_track_unit_vector_y'],
        args.flag_along_track, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['elevation_angle'],
        args.flag_elevation_angle, flag_all, cube_dataset_names_list)
    _update_cube_dataset_names_list(
        geometry_datasets_dict['ground_track_velocity'],
        args.flag_ground_track_velocity, flag_all, cube_dataset_names_list)

    # If list is empty, return
    if len(cube_dataset_names_list) == 0:
        return

    # Read interpolated DEM
    print(f'temporary file: {interpolated_dem_file}')

    dem_dataset = gdal.Open(interpolated_dem_file)
    if dem_dataset is None:
        print(f'ERROR opening file: {interpolated_dem_file}')
        return

    dem_band = dem_dataset.GetRasterBand(1)
    dem_image = dem_band.ReadAsArray()

    length = dem_dataset.RasterYSize
    width = dem_dataset.RasterXSize

    geotransform_output_file = dem_dataset.GetGeoTransform()
    dx = geotransform_output_file[1]
    dy = geotransform_output_file[5]
    x0 = geotransform_output_file[0] + 0.5 * dx
    y0 = geotransform_output_file[3] + 0.5 * dy
    xf = x0 + (width - 1) * dx
    yf = y0 + (length - 1) * dy

    vect_dtype = np.float64
    x_vect = np.tile(
        np.linspace(x0, xf, width, dtype=vect_dtype), length).ravel()
    y_vect = np.repeat(
        np.linspace(y0, yf, length, dtype=vect_dtype), width).ravel()
    z_vect = dem_image.ravel()

    # Create output coordinate vectors. y-vect is negated because
    # RegularGridInterpolator requires positive step
    points_stack = np.swapaxes(
        np.stack([z_vect, -y_vect, x_vect]), 0, 1)

    # Get metadata cube axis vectors
    product_metadata_cubes_path = f'{nisar_product_obj.MetadataPath}/radarGrid'
    metadata_cube_dataset = (f'{product_metadata_cubes_path}/' +
                             cube_dataset_names_list[0])
    metadata_cube_ref = f'NETCDF:{args.input_file}:{metadata_cube_dataset}'

    metadata_cube_ref_dataset = gdal.Open(metadata_cube_ref)
    if metadata_cube_ref_dataset is None:
        print(f'ERROR opening dataset "{metadata_cube_dataset}"'
              f' from file: {args.input_file}')
        return

    metadata = metadata_cube_ref_dataset.GetMetadata()
    cube_length = metadata_cube_ref_dataset.RasterYSize
    cube_width = metadata_cube_ref_dataset.RasterXSize

    geotransform_cube = metadata_cube_ref_dataset.GetGeoTransform()
    projection_cube = metadata_cube_ref_dataset.GetProjection()
    metadata_cube_ref_band = metadata_cube_ref_dataset.GetRasterBand(1)

    cube_dx = geotransform_cube[1]
    cube_dy = geotransform_cube[5]
    cube_x0 = geotransform_cube[0] + 0.5 * cube_dx
    cube_y0 = geotransform_cube[3] + 0.5 * cube_dy
    cube_xf = cube_x0 + (cube_width - 1) * cube_dx
    cube_yf = cube_y0 + (cube_length - 1) * cube_dy

    print('Input geogrid (cubes):')
    print('    length:', cube_length)
    print('    width:', cube_width)
    print('    x0, xf:', cube_x0, cube_xf)
    print('    y0, yf:', cube_y0, cube_yf)
    print('    dx:', cube_dx)
    print('    dy:', cube_dy)

    print('Positions to interpolate (1D):')
    print(f'    z size: {z_vect.size}, range: {[z_vect[0], z_vect[-1]]}')
    print(f'    y size: {y_vect.size}, range: {[y_vect[0], y_vect[-1]]}')
    print(f'    x size: {x_vect.size}, range: {[x_vect[0], x_vect[-1]]}')

    vect_dtype = np.float64

    x_vect = np.linspace(cube_x0, cube_xf, cube_width, dtype=vect_dtype)
    y_vect = np.linspace(cube_y0, cube_yf, cube_length, dtype=vect_dtype)

    values_str = metadata['NETCDF_DIM_heightAboveEllipsoid_VALUES']
    z_list = values_str.replace('{', '').replace('}', '').split(',')
    z_vect = np.asarray(z_list, dtype=vect_dtype)
    print(f'    z size: {z_vect.size}, range: {[z_vect[0], z_vect[-1]]}')
    print(f'    y size: {y_vect.size}, range: {[y_vect[0], y_vect[-1]]}')
    print(f'    x size: {x_vect.size}, range: {[x_vect[0], x_vect[-1]]}')

    # Create list of coordinates. y-vect is negated because
    # RegularGridInterpolator requires positive step
    vects = z_vect, -y_vect, x_vect

    print('Output geogrid:')
    print('    length:', length)
    print('    width:', width)
    print('    x0, xf:', x0, xf)
    print('    y0, yf:', y0, yf)
    print('    dx:', dx)
    print('    dy:', dy)

    for cube_dataset_name in cube_dataset_names_list:
        cube, geotransform_cube = \
            _load_cube(args.input_file, cube_dataset_name,
                       product_metadata_cubes_path)
        interp_function = RegularGridInterpolator(vects, cube,
                                                  bounds_error=False)
        interpolated_array = interp_function(points_stack)
        interpolated_2d_array = np.asarray(
            interpolated_array.reshape((length, width)))
        output_cube_file = os.path.join(args.output_dir,
                                        cube_dataset_name + '.tif')

        # save interpolated 2D array
        driver = gdal.GetDriverByName('GTiff')
        gdal_ds = driver.Create(output_cube_file, width, length,
                                nbands, metadata_cube_ref_band.DataType)
        gdal_band = gdal_ds.GetRasterBand(1)

        gdal_ds.SetGeoTransform(geotransform_output_file)
        gdal_ds.SetProjection(projection_cube)

        gdal_band.WriteArray(interpolated_2d_array)

        # flush updates to the disk
        gdal_band.FlushCache()
        gdal_band = None
        gdal_ds = None

        print(f'file saved: {output_cube_file}')

    if not args.flag_interpolated_dem:
        os.remove(interpolated_dem_file)
    else:
        print(f'file saved: {interpolated_dem_file}')


def _load_cube(input_file, dataset_name, product_metadata_cubes_path):

    dataset = f'{product_metadata_cubes_path}/{dataset_name}'
    dataset_name = f'NETCDF:{input_file}:{dataset}'
    gdal_ds = gdal.Open(dataset_name)
    geotransform_cube = gdal_ds.GetGeoTransform()
    image_list = []
    for b in range(gdal_ds.RasterCount):
        band = gdal_ds.GetRasterBand(b + 1)
        image = band.ReadAsArray()
        image_list.append(image)
    cube = np.stack(image_list)
    return cube, geotransform_cube


def _update_cube_dataset_names_list(dataset, flag_create_dataset,
                                    flag_all, cube_dataset_names_list):
    if not flag_create_dataset and not flag_all:
        return
    cube_dataset_names_list.append(dataset)


def _get_raster(output_dir, ds_name, dtype, shape,
                output_file_list, output_obj_list):
    """Create an ISCE3 raster object (GTiff) for a radar geometry layer.

       Parameters
       ----------
       output_dir: str
              Output directory
       ds_name: str
              Dataset (geometry layer) name
       dtype:: gdal.DataType
              GDAL data type
       shape: list
              Shape of the output raster
       output_file_list: list
              Mutable list of output files
       output_obj_list: list
              Mutable list of output raster objects

       Returns
       -------
       raster_obj : isce3.io.Raster
              ISCE3 raster object
       output_file : str
              Output raster file name
    """
    output_file = os.path.join(output_dir, ds_name)+'.tif'
    raster_obj = isce3.io.Raster(
        output_file,
        shape[2],
        shape[1],
        shape[0],
        dtype,
        "GTiff")
    output_file_list.append(output_file)
    output_obj_list.append(raster_obj)
    return raster_obj, output_file


def main(argv=None):
    argv = get_parser()
    run(argv)


if __name__ == '__main__':
    main()
