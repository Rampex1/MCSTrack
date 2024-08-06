import asyncio
import pyigtl
import time as t
import logging

from src.common.api.mct_request_series import MCTRequestSeries
from src.common.structures.component_role_label import COMPONENT_ROLE_LABEL_DETECTOR, COMPONENT_ROLE_LABEL_POSE_SOLVER
from src.controller.mct_controller import MCTController
from src.controller.structures.mct_component_address import MCTComponentAddress
from ipaddress import IPv4Address

from src.pose_solver.api import PoseSolverAddTargetMarkerRequest
from src.pose_solver.api import PoseSolverSetReferenceRequest

from src.controller.structures.connection import Connection
from src.pose_solver.structures import TargetMarker

# TODO: these setup-specific lines need to be updated/removed
# Would be better to pass a configuration to the controller
pose_solver_label = "p"
detector_labels_and_IPs = [
    ("d101",IPv4Address('192.168.0.101')),
    ("d102",IPv4Address('192.168.0.102'))]

server = pyigtl.OpenIGTLinkServer(port=18944,local_server=False)

async def update(controller):
    try:
        await controller.update()
    except Exception as e:
        controller.add_status_message(
            severity="error",
            message=f"Exception occurred in controller loop: {str(e)}")
        
    ps_frame = controller.get_live_pose_solver_frame(pose_solver_label)
    timestamp = ps_frame.timestamp_utc_iso8601

    if len(ps_frame.target_poses) > 0:
        target_poses = ps_frame.target_poses[0]
        message = pyigtl.TransformMessage(
            matrix=target_poses.object_to_reference_matrix.as_numpy_array(),
            # timestamp=target_poses.solver_timestamp_utc_iso8601,
            device_name=target_poses.target_id
        )
        server.send_message(message,wait=True)

    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(update(controller))

async def main():
    logging.basicConfig(level=logging.INFO)

    controller = MCTController(
        serial_identifier="controller",
        send_status_messages_to_logger=True)
    
    controller.add_status_message(
            severity="info",
            message=f"Slicer client connected")

    # TODO: this assumes pose solver is running on the same computer
    # Should be updated/removed and instead determined by a config file passed to controller
    controller.add_connection(MCTComponentAddress(
            label=pose_solver_label,
            role=COMPONENT_ROLE_LABEL_POSE_SOLVER,
            ip_address=IPv4Address("127.0.0.1"),
            port=8000
        ))
    
    for (label, ip) in detector_labels_and_IPs:
        controller.add_connection(MCTComponentAddress(
                label=label,
                role=COMPONENT_ROLE_LABEL_DETECTOR,
                ip_address=ip,
                port=8001
            ))

    controller.start_up()

    while controller.is_transitioning():
        await controller.update()

    # TODO: these setup-specific lines need to be updated/removed
    # Tracked and reference marker should be set via controller config file
    # Set reference marker for pose solver
    request_series: MCTRequestSeries = MCTRequestSeries(series=[
        PoseSolverSetReferenceRequest(
                    marker_id=1,
                    marker_diameter=10)])
    controller.request_series_push(
        connection_label=pose_solver_label,
        request_series=request_series)

    # Set tracked marker for pose solver
    request_series: MCTRequestSeries = MCTRequestSeries(series=[
        PoseSolverAddTargetMarkerRequest(
                target=TargetMarker(
                    target_id=0,
                    marker_id=0,
                    marker_size=10))])
    controller.request_series_push(
        connection_label=pose_solver_label,
        request_series=request_series)

    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
    event_loop.create_task(update(controller=controller))

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    while not server.is_connected():
        t.sleep(0.1)

    asyncio.run(main())
    