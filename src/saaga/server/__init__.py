"""SAAGA Server subpackage."""
# Lazy import app to keep file_manager and normalizers usable without fastapi/uvicorn
__all__ = ["get_app"]

def get_app():
    from saaga.server.app import app
    return app
