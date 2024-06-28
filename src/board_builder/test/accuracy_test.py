from src.board_builder.board_builder import BoardBuilder
import src.board_builder.test.data as data
import numpy as np


class AccuracyTest:
    def __init__(self, reference_data, collection_data, building_data, theoretical_data):
        self.reference_data = reference_data
        self.collection_data = collection_data
        self.building_data = building_data
        self.theoretical_data = theoretical_data
        self.board_builder = BoardBuilder()

    def _compare_data_sets(self, theoretical_data, experimental_data):
        if len(theoretical_data) != len(experimental_data):
            return False

        sorted_theoretical = sorted(theoretical_data, key=lambda x: np.sum(x))
        sorted_experimental = sorted(experimental_data, key=lambda x: np.sum(x))

        for arr1, arr2 in zip(sorted_theoretical, sorted_experimental):
            if not np.allclose(arr1, arr2):
                return False
        return True

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

    def run_accuracy_test(self):
        self.board_builder.pose_solver.set_board_marker_size(10)

        # Reference Data
        for detector_name in self.reference_data:
            self.board_builder.pose_solver.set_intrinsic_parameters(detector_name, self.reference_data[detector_name]["intrinsics"])
        self.board_builder.locate_reference_markers(self.reference_data)

        # Collect Data
        self.board_builder.collect_data(self.collection_data)

        # Build Board
        corners_dict2 = self.board_builder.build_board(self.building_data)

        # RMS Error
        corners_list = []
        for uuid in corners_dict2:
            corners_list.append(corners_dict2[uuid])

        rms_error = self._calculate_rms_error(corners_list, self.theoretical_data)
        print(f"RMS Error: {rms_error}")

accuracy_tester = AccuracyTest(data.REFERENCE_DATA, data.COLLECTION_DATA, data.BUILDING_DATA, data.THEORETICAL_DATA)
accuracy_tester.run_accuracy_test()
