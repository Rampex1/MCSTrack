import datetime
import numpy as np
from src.board_builder.board_builder import BoardBuilder
import src.board_builder.test.data as data


class AccuracyTest:
    def __init__(self, reference_data):
        self.reference_data = reference_data

    @staticmethod
    def compare_matrices(matrix1, matrix2):
        return np.allclose(matrix1, matrix2, atol=1e-6)

    @staticmethod
    def evaluate_accuracy(estimated_matrix, theoretical_matrix):
        if len(estimated_matrix) != len(theoretical_matrix):
            return False
        for i in range(len(estimated_matrix)):
            for j in range(len(estimated_matrix[i])):
                if theoretical_matrix[i][j] is None:
                    continue
                if not AccuracyTest.compare_matrices(estimated_matrix[i][j].get_matrix(), theoretical_matrix[i][j].get_matrix()):
                    return False
        return True

    def run_accuracy_test(self):
        board_builder = BoardBuilder()
        for detector_name in self.reference_data:
            board_builder.pose_solver.set_intrinsic_parameters(detector_name, self.reference_data[detector_name]["intrinsics"])
        board_builder.locate_reference_markers(self.reference_data)

        # Collect Data

        # Build Board

        accuracy = True
        print(f"Accuracy: {'Pass' if accuracy else 'Fail'}")


accuracy_tester = AccuracyTest(data.REFERENCE_DATA)
accuracy_tester.run_accuracy_test()
