import numpy as np
from src.board_builder.board_builder import BoardBuilder

from src.board_builder.test.Lab import lab_data

class AccuracyTest:
    ACCURACY_PERCENTAGE = 75  # Accuracy of the corners data
    NOISE_LEVEL = 2.0  # Define noise level of inaccurate corner data

    def __init__(self, detector_poses, collection_data, building_data, theoretical_data):
        self.detector_poses = detector_poses
        self.collection_data = collection_data
        self.building_data = building_data
        self.theoretical_data = theoretical_data
        self.board_builder_experimental = BoardBuilder()
        self.board_builder_theoretical = BoardBuilder()

    def _add_noise_to_corners(self, data, noise_level):
        noisy_data = {}
        for detector_name, detector_data in data.items():
            noisy_corners = []
            for corner in detector_data['corners']:
                noisy_corner = corner.copy()
                total_points = corner.size
                num_noisy_points = int((1 - self.ACCURACY_PERCENTAGE / 100) * total_points)
                indices = np.random.choice(total_points, num_noisy_points, replace=False)
                for index in indices:
                    noise = np.random.uniform(-noise_level, noise_level)
                    noisy_corner.flat[index] += noise
                noisy_corner = np.round(noisy_corner)  # Ensure values are integers
                noisy_corners.append(noisy_corner)
            noisy_data[detector_name] = {
                'ids': detector_data['ids'],
                'corners': noisy_corners,
                'intrinsics': detector_data['intrinsics']
            }
        return noisy_data

    def _calculate_rms_error(self, theoretical_data, experimental_data):
        if len(theoretical_data) != len(experimental_data):
            raise ValueError("Theoretical and experimental data must have the same length.")

        rms_error = 0.0
        total_points = 0

        for arr1, arr2 in zip(theoretical_data, experimental_data):
            if arr1.shape != arr2.shape:
                raise ValueError("Corresponding arrays must have the same shape.")
            diff = arr1 - arr2
            rms_error += np.sum(diff ** 2)
            total_points += arr1.size

        rms_error = np.sqrt(rms_error / total_points)
        return rms_error

    def _get_corners(self, corners_dict):
        corners = []
        for uuid in corners_dict:
            corners.append(corners_dict[uuid])
        return corners

    def run_accuracy_test(self):
        self.board_builder_experimental.pose_solver.set_board_marker_size(10)
        self.board_builder_theoretical.pose_solver.set_board_marker_size(10)

        # Reference Data (Enter detector poses)
        for detector_name in self.collection_data:
            self.board_builder_experimental.pose_solver.set_intrinsic_parameters(detector_name, self.collection_data[detector_name]["intrinsics"])
            self.board_builder_theoretical.pose_solver.set_intrinsic_parameters(detector_name, self.collection_data[detector_name]["intrinsics"])
        self.board_builder_experimental.pose_solver.set_detector_poses(self.detector_poses)
        self.board_builder_theoretical.pose_solver.set_detector_poses(self.detector_poses)

        # Add noise to the collection and building data
        noisy_collection_data = self._add_noise_to_corners(self.collection_data, self.NOISE_LEVEL)
        noisy_building_data = self._add_noise_to_corners(self.building_data, self.NOISE_LEVEL)
        clean_collection_data = self._add_noise_to_corners(self.collection_data, 0)
        clean_building_data = self._add_noise_to_corners(self.building_data, 0)

        # Collect Data
        self.board_builder_experimental.collect_data(noisy_collection_data)
        self.board_builder_theoretical.collect_data(clean_collection_data)

        # Build Board
        corners_dict_experimental = self.board_builder_experimental.build_board(noisy_building_data)
        corners_dict_theoretical = self.board_builder_theoretical.build_board(clean_building_data)

        # RMS Error
        rms_error = self._calculate_rms_error(self._get_corners(corners_dict_theoretical), self._get_corners(corners_dict_experimental))
        return rms_error


RUN_TIME = 10
NAME = "lab_data"

accuracy_tester = AccuracyTest(lab_data.DETECTOR_POSES, lab_data.COLLECTION_DATA, lab_data.BUILDING_DATA, lab_data.THEORETICAL_DATA)
rms_errors = [accuracy_tester.run_accuracy_test() for _ in range(RUN_TIME)]

average_rms = np.mean(rms_errors)
median_rms = np.median(rms_errors)

print(f"Average RMS Error: {average_rms}")
print(f"Median RMS Error: {median_rms}")
