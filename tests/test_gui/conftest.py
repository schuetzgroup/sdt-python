import pytest


# need to set application and organization name for QSettings
# pytest-qt only sets the former
@pytest.fixture(scope="session", autouse=True)
def config_app_settings(qapp):
    qapp.setOrganizationName("pytest-sdt-gui-org")
    qapp.setApplicationName("pytest-sdt-gui-app")
