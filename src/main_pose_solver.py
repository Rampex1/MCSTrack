import logging
import uvicorn


def main():
    uvicorn.run(
        "src.pose_solver.pose_solver_app:app",
        reload=False,
        port=8000,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
