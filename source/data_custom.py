import pandas as pd

from os import path as osp
from time_utils import custom_timestamp_sec_parser, custom_timestamp_nano_parser


class CustomData:
    _DATA_TIMESTAMP = 'DATA_TIMESTAMP'
    _PHONE_TIMESTAMP = 'PHONE_TIMESTAMP'
    _PHONE_ACCELEROMETER_X = 'PHONE_ACCELEROMETER_X'
    _PHONE_ACCELEROMETER_Y = 'PHONE_ACCELEROMETER_Y'
    _PHONE_ACCELEROMETER_Z = 'PHONE_ACCELEROMETER_Z'
    _PHONE_GYROSCOPE_X = 'PHONE_GYROSCOPE_X'
    _PHONE_GYROSCOPE_Y = 'PHONE_GYROSCOPE_Y'
    _PHONE_GYROSCOPE_Z = 'PHONE_GYROSCOPE_Z'
    _PHONE_GAME_ROTATION_VECTOR_X = 'PHONE_GAME_ROTATION_VECTOR_X'
    _PHONE_GAME_ROTATION_VECTOR_Y = 'PHONE_GAME_ROTATION_VECTOR_Y'
    _PHONE_GAME_ROTATION_VECTOR_Z = 'PHONE_GAME_ROTATION_VECTOR_Z'
    _PHONE_GAME_ROTATION_VECTOR_SCALAR = 'PHONE_GAME_ROTATION_VECTOR_SCALAR'
    _PHONE_ROTATION_VECTOR_X = 'PHONE_ROTATION_VECTOR_X'
    _PHONE_ROTATION_VECTOR_Y = 'PHONE_ROTATION_VECTOR_Y'
    _PHONE_ROTATION_VECTOR_Z = 'PHONE_ROTATION_VECTOR_Z'
    _PHONE_ROTATION_VECTOR_SCALAR = 'PHONE_ROTATION_VECTOR_SCALAR'
    _PHONE_GYROSCOPE_UNCALIBRATED_X = 'PHONE_GYROSCOPE_UNCALIBRATED_X'
    _PHONE_GYROSCOPE_UNCALIBRATED_Y = 'PHONE_GYROSCOPE_UNCALIBRATED_Y'
    _PHONE_GYROSCOPE_UNCALIBRATED_Z = 'PHONE_GYROSCOPE_UNCALIBRATED_Z'
    _PHONE_ORIENTATION_X = 'PHONE_ORIENTATION_X'
    _PHONE_ORIENTATION_Y = 'PHONE_ORIENTATION_Y'
    _PHONE_ORIENTATION_Z = 'PHONE_ORIENTATION_Z'
    _PHONE_MAGNETIC_FIELD_X = 'PHONE_MAGNETIC_FIELD_X'
    _PHONE_MAGNETIC_FIELD_Y = 'PHONE_MAGNETIC_FIELD_Y'
    _PHONE_MAGNETIC_FIELD_Z = 'PHONE_MAGNETIC_FIELD_Z'
    _PHONE_GRAVITY_X = 'PHONE_GRAVITY_X'
    _PHONE_GRAVITY_Y = 'PHONE_GRAVITY_Y'
    _PHONE_GRAVITY_Z = 'PHONE_GRAVITY_Z'
    _PHONE_LINEAR_ACCELERATION_X = 'PHONE_LINEAR_ACCELERATION_X'
    _PHONE_LINEAR_ACCELERATION_Y = 'PHONE_LINEAR_ACCELERATION_Y'
    _PHONE_LINEAR_ACCELERATION_Z = 'PHONE_LINEAR_ACCELERATION_Z'
    _PHONE_PRESSURE = 'PHONE_PRESSURE'
    _PHONE_GNSS_TIMESTAMP = 'PHONE_GNSS_TIMESTAMP'
    _PHONE_GNSS_LONGITUDE = 'PHONE_GNSS_LONGITUDE'
    _PHONE_GNSS_LATITUDE = 'PHONE_GNSS_LATITUDE'
    _PHONE_GNSS_ACCURACY = 'PHONE_GNSS_ACCURACY'
    _PHONE_ALKAID_REQUEST_TIMESTAMP = 'PHONE_ALKAID_REQUEST_TIMESTAMP'
    _PHONE_ALKAID_RESPONSIVE_TIMESTAMP = 'PHONE_ALKAID_RESPONSIVE_TIMESTAMP'
    _PHONE_ALKAID_MAP_TIMESTAMP = 'PHONE_ALKAID_MAP_TIMESTAMP'
    _PHONE_ALKAID_STATUS = 'PHONE_ALKAID_STATUS'
    _PHONE_ALKAID_TIMESTAMP = 'PHONE_ALKAID_TIMESTAMP'
    _PHONE_ALKAID_SYNCHRONIZE_TIMESTAMP = 'PHONE_ALKAID_SYNCHRONIZE_TIMESTAMP'
    _PHONE_ALKAID_LONGITUDE = 'PHONE_ALKAID_LONGITUDE'
    _PHONE_ALKAID_LATITUDE = 'PHONE_ALKAID_LATITUDE'
    _PHONE_ALKAID_HEIGHT = 'PHONE_ALKAID_HEIGHT'
    _PHONE_ALKAID_AZIMUTH = 'PHONE_ALKAID_AZIMUTH'
    _PHONE_ALKAID_SPEED = 'PHONE_ALKAID_SPEED'
    _ALKAID_TIMESTAMP = '_ALKAID_TIMESTAMP'
    _ALKAID_LONGITUDE = 'ALKAID_LONGITUDE'
    _ALKAID_LATITUDE = 'ALKAID_LATITUDE'
    _ALKAID_PROJECT_COORDINATE_X = 'ALKAID_PROJECT_COORDINATE_X'
    _ALKAID_PROJECT_COORDINATE_Y = 'ALKAID_PROJECT_COORDINATE_Y'
    _ALKAID_PROJECT_COORDINATE_DELTA_X = 'ALKAID_PROJECT_COORDINATE_DELTA_X'
    _ALKAID_PROJECT_COORDINATE_DELTA_Y = 'ALKAID_PROJECT_COORDINATE_DELTA_Y'
    _ALKAID_HEIGHT = 'ALKAID_HEIGHT'
    _ALKAID_AZIMUTH = 'ALKAID_AZIMUTH'
    _ALKAID_SPEED = 'ALKAID_SPEED'
    _ALKAID_AGE = 'ALKAID_AGE'

    _CUSTOM_DATA_NAMES_LIST = [
        _DATA_TIMESTAMP,
        _PHONE_ACCELEROMETER_X,
        _PHONE_ACCELEROMETER_Y,
        _PHONE_ACCELEROMETER_Z,
        _PHONE_GYROSCOPE_X,
        _PHONE_GYROSCOPE_Y,
        _PHONE_GYROSCOPE_Z,
        _PHONE_GAME_ROTATION_VECTOR_X,
        _PHONE_GAME_ROTATION_VECTOR_Y,
        _PHONE_GAME_ROTATION_VECTOR_Z,
        _PHONE_GAME_ROTATION_VECTOR_SCALAR,
        _PHONE_ROTATION_VECTOR_X,
        _PHONE_ROTATION_VECTOR_Y,
        _PHONE_ROTATION_VECTOR_Z,
        _PHONE_ROTATION_VECTOR_SCALAR,
        _PHONE_GYROSCOPE_UNCALIBRATED_X,
        _PHONE_GYROSCOPE_UNCALIBRATED_Y,
        _PHONE_GYROSCOPE_UNCALIBRATED_Z,
        _PHONE_ORIENTATION_X,
        _PHONE_ORIENTATION_Y,
        _PHONE_ORIENTATION_Z,
        _PHONE_MAGNETIC_FIELD_X,
        _PHONE_MAGNETIC_FIELD_Y,
        _PHONE_MAGNETIC_FIELD_Z,
        _PHONE_GRAVITY_X,
        _PHONE_GRAVITY_Y,
        _PHONE_GRAVITY_Z,
        _PHONE_LINEAR_ACCELERATION_X,
        _PHONE_LINEAR_ACCELERATION_Y,
        _PHONE_LINEAR_ACCELERATION_Z,
        _PHONE_PRESSURE,
        _PHONE_GNSS_TIMESTAMP,
        _PHONE_GNSS_LONGITUDE,
        _PHONE_GNSS_LATITUDE,
        _PHONE_GNSS_ACCURACY,
        _PHONE_ALKAID_REQUEST_TIMESTAMP,
        _PHONE_ALKAID_RESPONSIVE_TIMESTAMP,
        _PHONE_ALKAID_MAP_TIMESTAMP,
        _PHONE_ALKAID_STATUS,
        _PHONE_ALKAID_TIMESTAMP,
        _PHONE_ALKAID_SYNCHRONIZE_TIMESTAMP,
        _PHONE_ALKAID_LONGITUDE,
        _PHONE_ALKAID_LATITUDE,
        _PHONE_ALKAID_HEIGHT,
        _PHONE_ALKAID_AZIMUTH,
        _PHONE_ALKAID_SPEED,
        _ALKAID_TIMESTAMP,
        _ALKAID_LONGITUDE,
        _ALKAID_LATITUDE,
        _ALKAID_PROJECT_COORDINATE_X,
        _ALKAID_PROJECT_COORDINATE_Y,
        _ALKAID_PROJECT_COORDINATE_DELTA_X,
        _ALKAID_PROJECT_COORDINATE_DELTA_Y,
        _ALKAID_HEIGHT,
        _ALKAID_AZIMUTH,
        _ALKAID_SPEED,
        _ALKAID_AGE
    ]

    @staticmethod
    def parse(folder, source_all):
        path = osp.join(folder, 'trainVdrExperimentTimeTable.txt')
        custom_raw_data = pd.read_csv(
            path,
            header=0,
            names=CustomData._CUSTOM_DATA_NAMES_LIST
        )
        custom_parse_data = custom_raw_data.copy(deep=True)
        custom_parse_data[CustomData._DATA_TIMESTAMP] = custom_parse_data[CustomData._DATA_TIMESTAMP] \
            .map(custom_timestamp_sec_parser)

        custom_write_data = custom_raw_data.copy(deep=True)
        custom_write_data[CustomData._DATA_TIMESTAMP] = custom_write_data[CustomData._DATA_TIMESTAMP] \
            .map(custom_timestamp_nano_parser)

        all_sources = {}
        for source in source_all:
            if source == 'gyro':
                source_gyro = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_GYROSCOPE_X,
                               CustomData._PHONE_GYROSCOPE_Y,
                               CustomData._PHONE_GYROSCOPE_Z]
                              ]
                all_sources['gyro'] = source_gyro.to_numpy()

                source_gyro_write = custom_write_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_GYROSCOPE_X,
                               CustomData._PHONE_GYROSCOPE_Y,
                               CustomData._PHONE_GYROSCOPE_Z]
                              ]
                source_gyro_write_path = osp.join(folder, 'gyro.txt')
                source_gyro_write.to_csv(source_gyro_write_path, sep=' ', header=False, index=False)

            if source == 'gyro_uncalib':
                source_gyro_uncalib = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_GYROSCOPE_UNCALIBRATED_X,
                               CustomData._PHONE_GYROSCOPE_UNCALIBRATED_Y,
                               CustomData._PHONE_GYROSCOPE_UNCALIBRATED_Z]
                              ]
                all_sources['gyro_uncalib'] = source_gyro_uncalib.to_numpy()

                source_gyro_uncalib_write = custom_write_data.loc[
                                      :,
                                      [CustomData._DATA_TIMESTAMP,
                                       CustomData._PHONE_GYROSCOPE_UNCALIBRATED_X,
                                       CustomData._PHONE_GYROSCOPE_UNCALIBRATED_Y,
                                       CustomData._PHONE_GYROSCOPE_UNCALIBRATED_Z]
                                      ]
                source_gyro_uncalib_write_path = osp.join(folder, 'gyro_uncalib.txt')
                source_gyro_uncalib_write.to_csv(source_gyro_uncalib_write_path, sep=' ', header=False, index=False)

            if source == 'acce':
                source_acce = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_ACCELEROMETER_X,
                               CustomData._PHONE_ACCELEROMETER_Y,
                               CustomData._PHONE_ACCELEROMETER_Z]
                              ]
                all_sources['acce'] = source_acce.to_numpy()

                source_acce_write = custom_write_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_ACCELEROMETER_X,
                               CustomData._PHONE_ACCELEROMETER_Y,
                               CustomData._PHONE_ACCELEROMETER_Z]
                              ]
                source_acce_write_path = osp.join(folder, 'acce.txt')
                source_acce_write.to_csv(source_acce_write_path, sep=' ', header=False, index=False)

            if source == 'linacce':
                source_linacce = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_LINEAR_ACCELERATION_X,
                               CustomData._PHONE_LINEAR_ACCELERATION_Y,
                               CustomData._PHONE_LINEAR_ACCELERATION_Z]
                              ]
                all_sources['linacce'] = source_linacce.to_numpy()

                source_linacce_write = custom_write_data.loc[
                                 :,
                                 [CustomData._DATA_TIMESTAMP,
                                  CustomData._PHONE_LINEAR_ACCELERATION_X,
                                  CustomData._PHONE_LINEAR_ACCELERATION_Y,
                                  CustomData._PHONE_LINEAR_ACCELERATION_Z]
                                 ]
                source_linacce_write_path = osp.join(folder, 'linacce.txt')
                source_linacce_write.to_csv(source_linacce_write_path, sep=' ', header=False, index=False)

            if source == 'gravity':
                source_gravity = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_GRAVITY_X,
                               CustomData._PHONE_GRAVITY_Y,
                               CustomData._PHONE_GRAVITY_Z]
                              ]
                all_sources['gravity'] = source_gravity.to_numpy()

                source_gravity_write = custom_write_data.loc[
                                 :,
                                 [CustomData._DATA_TIMESTAMP,
                                  CustomData._PHONE_GRAVITY_X,
                                  CustomData._PHONE_GRAVITY_Y,
                                  CustomData._PHONE_GRAVITY_Z]
                                 ]
                source_gravity_write_path = osp.join(folder, 'gravity.txt')
                source_gravity_write.to_csv(source_gravity_write_path, sep=' ', header=False, index=False)

            if source == 'magnet':
                source_magnet = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_MAGNETIC_FIELD_X,
                               CustomData._PHONE_MAGNETIC_FIELD_Y,
                               CustomData._PHONE_MAGNETIC_FIELD_Z]
                              ]
                all_sources['magnet'] = source_magnet.to_numpy()

                source_magnet_write = custom_write_data.loc[
                                :,
                                [CustomData._DATA_TIMESTAMP,
                                 CustomData._PHONE_MAGNETIC_FIELD_X,
                                 CustomData._PHONE_MAGNETIC_FIELD_Y,
                                 CustomData._PHONE_MAGNETIC_FIELD_Z]
                                ]
                source_magnet_write_path = osp.join(folder, 'magnet.txt')
                source_magnet_write.to_csv(source_magnet_write_path, sep=' ', header=False, index=False)

            if source == 'rv':
                source_rv = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_ROTATION_VECTOR_X,
                               CustomData._PHONE_ROTATION_VECTOR_Y,
                               CustomData._PHONE_ROTATION_VECTOR_Z,
                               CustomData._PHONE_ROTATION_VECTOR_SCALAR]
                              ]
                all_sources['rv'] = source_rv.to_numpy()

                source_rv_write = custom_write_data.loc[
                            :,
                            [CustomData._DATA_TIMESTAMP,
                             CustomData._PHONE_ROTATION_VECTOR_X,
                             CustomData._PHONE_ROTATION_VECTOR_Y,
                             CustomData._PHONE_ROTATION_VECTOR_Z,
                             CustomData._PHONE_ROTATION_VECTOR_SCALAR]
                            ]
                source_rv_write_path = osp.join(folder, 'rv.txt')
                source_rv_write.to_csv(source_rv_write_path, sep=' ', header=False, index=False)

            if source == 'game_rv':
                source_game_rv = custom_parse_data.loc[
                              :,
                              [CustomData._DATA_TIMESTAMP,
                               CustomData._PHONE_GAME_ROTATION_VECTOR_X,
                               CustomData._PHONE_GAME_ROTATION_VECTOR_Y,
                               CustomData._PHONE_GAME_ROTATION_VECTOR_Z,
                               CustomData._PHONE_GAME_ROTATION_VECTOR_SCALAR]
                              ]
                all_sources['game_rv'] = source_game_rv.to_numpy()

                source_game_rv_write = custom_write_data.loc[
                                 :,
                                 [CustomData._DATA_TIMESTAMP,
                                  CustomData._PHONE_GAME_ROTATION_VECTOR_X,
                                  CustomData._PHONE_GAME_ROTATION_VECTOR_Y,
                                  CustomData._PHONE_GAME_ROTATION_VECTOR_Z,
                                  CustomData._PHONE_GAME_ROTATION_VECTOR_SCALAR]
                                 ]
                source_game_rv_write_path = osp.join(folder, 'game_rv.txt')
                source_game_rv_write.to_csv(source_game_rv_write_path, sep=' ', header=False, index=False)

        return all_sources
