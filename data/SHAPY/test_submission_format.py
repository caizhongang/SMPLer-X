""" Ref: https://github.com/muelea/shapy/blob/master/regressor/hbw_evaluation/test_submission_format.py """

import argparse
import numpy as np


def test_submission_file_format(
        npz_file: str,
        model_type: str = 'smplx'
):
    submission = np.load(npz_file)

    # check if keys are named correctly
    keys = [x for x in submission.keys()]
    assert 'image_name' in keys and 'v_shaped' in keys, \
        f"Keys are not correct. Got {keys}, but expected ['image_name', 'v_shaped']"

    image_names = submission['image_name']
    v_shapeds = submission['v_shaped']

    # check if shape and type are correct
    assert type(image_names) == np.ndarray, \
        f"Type of key image_name is not correct. {type(image_names)} given, but np.ndarray expected."
    assert image_names.shape == (1631,), \
        f"Shape of key image_name is not correct. {image_names.shape} given, but (1631,) expected."

    assert type(v_shapeds) == np.ndarray, \
        f"Type of key v_shaped is not correct. {type(image_names)} given, but np.ndarray expected."

    if model_type == 'smplx':
        assert v_shapeds.shape == (1631, 10475, 3), \
            f"Shape of key v_shaped is not correct. {v_shapeds.shape} given, but (1631, 10475, 3) expected."
    else:
        assert v_shapeds.shape == (1631, 6890, 3), \
            f"Shape of key v_shaped is not correct. {v_shapeds.shape} given, but (1631, 6890, 3) expected."

    # check if each image has a prediction
    hbw_images_gt = np.load('../data/SHAPY/hbw_testset_image_names.npy')
    check_prediction_available = np.isin(hbw_images_gt, image_names)
    assert np.all(check_prediction_available), \
        f"Images without predition exist! Missing predictions: \
            \n {hbw_images_gt[~check_prediction_available]}"

    print(f'Your submission file passed the test.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-npz-file',
                        dest='input_npz_file', type=str, required=True,
                        help='npz containing labels and body shape parameters.')
    parser.add_argument('--model-type', choices=['smpl', 'smplx'], type=str,
                        default='smplx',
                        help='The model type used for body shape prediction. ')

    args = parser.parse_args()

    test_submission_file_format(
        npz_file=args.input_npz_file,
        model_type=args.model_type
    )
