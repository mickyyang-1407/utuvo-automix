"""Shared test helpers."""
import pytest


@pytest.fixture
def schema_free(tmp_path_factory):
    """A schema that allows any plan, for ``plan.validate(p, schema_free)``.

    The shipped schema is mandatory and runs first, so it refuses most bad
    plans before the engine sees them.  Validating against this schema
    leaves only the engine's own checks, which must still refuse them.
    """
    path = tmp_path_factory.mktemp('schema') / 'any.schema.json'
    path.write_text('{}', encoding='utf-8')
    return str(path)
