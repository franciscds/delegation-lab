import pytest
from fastapi.testclient import TestClient

from delegation_lab.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
