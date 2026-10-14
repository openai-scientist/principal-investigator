import uvicorn

from principal_investigator.api.app import create_app
from principal_investigator.settings import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
