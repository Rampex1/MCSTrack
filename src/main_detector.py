from src.detector.detector_app import app
import logging
import uvicorn


def main():
    uvicorn.run(
        app,
        reload=False,
        port=8001,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
