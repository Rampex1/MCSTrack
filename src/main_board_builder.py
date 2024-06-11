import logging
import uvicorn


def main():
    uvicorn.run(
        "board_builder.board_builder_app:app",
        reload=False,
        port=7998,
        log_level=logging.INFO)


if __name__ == "__main__":
    main()
