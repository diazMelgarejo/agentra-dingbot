# Playwright UI tests run in a separate process from the asyncio test suite.
# pytest-asyncio's auto mode conflicts with Playwright's sync event loop;
# keeping them isolated here avoids that conflict.
# Run via: python -m pytest tests/ui/ --no-header  OR  make test-ui
