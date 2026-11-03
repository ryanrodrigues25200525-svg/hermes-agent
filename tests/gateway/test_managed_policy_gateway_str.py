"""managed_policy() must not crash on profile configs.

Regression for #148585: profile configs carry ``gateway: running`` (a status
string) where the global config carries the ``gateway:`` mapping, and the
chained ``.get('gateway', {}).get(...)`` lookup crashed with AttributeError,
breaking turn handling in the desktop app.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock


def _authority():
    source = SimpleNamespace()
    session = SimpleNamespace(source=source)
    return SimpleNamespace(runner=object(), sessions={"sess": session})


def _policy(config):
    return SimpleNamespace(
        ignore_user_config=False,
        kanban_json=None,
        config=lambda: config,
        request_json=json.dumps({"provider": "custom", "base_url": "http://x"}),
        platform="cli",
    )


class ManagedPolicyGatewayShapeTests(unittest.TestCase):
    def _managed_policy(self, agent_policy):
        from gateway import session_managed_worker as worker

        ref = SimpleNamespace(session_id="sess")
        with mock.patch(
            "gateway.session_policy.policy_for_source", return_value=agent_policy
        ):
            return worker.managed_policy(_authority(), ref)

    def test_string_gateway_fails_closed(self) -> None:
        self.assertIsNone(self._managed_policy(_policy({"gateway": "running"})))

    def test_missing_gateway_fails_closed(self) -> None:
        self.assertIsNone(self._managed_policy(_policy({})))

    def test_non_dict_config_fails_closed(self) -> None:
        self.assertIsNone(self._managed_policy(_policy(None)))

    def test_mapping_gateway_opts_in(self) -> None:
        policy = _policy({"gateway": {"managed_workers": True}})
        self.assertIs(policy, self._managed_policy(policy))

    def test_explicit_false_opts_out(self) -> None:
        policy = _policy({"gateway": {"managed_workers": False}})
        self.assertIsNone(self._managed_policy(policy))


if __name__ == "__main__":
    unittest.main()
