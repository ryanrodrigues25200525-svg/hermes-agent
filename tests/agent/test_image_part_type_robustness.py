"""Image-type checks must not crash on unhashable ``type`` values.

Regression for #148586: structured tool results can carry JSON Schema
union types such as ``{"type": ["string", "null"]}``. Every ``type in
{...}`` membership test over such values raised ``TypeError`` before the
provider request opened. The same crash class existed in three places;
all three are pinned here.
"""

import unittest

from agent.anthropic_message_convert import _convert_content_part_to_anthropic
from agent.client_lifecycle import ClientLifecycleMixin
from agent.conversation_compression import try_shrink_image_parts_in_messages

_UNHASHABLE_TYPES = [
    ["string", "null"],
    [],
    {"format": "uri"},
    None,
    42,
    "text",
]


def _kwargs(field, type_value):
    return {
        field: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": {"schema": {"type": type_value}},
                    }
                ],
            }
        ]
    }


class ImagePartTypeRobustnessTests(unittest.TestCase):
    def test_outbound_scan_ignores_unhashable_types(self) -> None:
        for field in ("messages", "input"):
            for type_value in _UNHASHABLE_TYPES:
                with self.subTest(field=field, type_value=type_value):
                    self.assertFalse(
                        ClientLifecycleMixin._api_kwargs_have_image_parts(
                            _kwargs(field, type_value)
                        )
                    )

    def test_outbound_scan_still_detects_real_images(self) -> None:
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:x"}},
                        {"type": ["string", "null"]},
                    ],
                }
            ]
        }
        self.assertTrue(ClientLifecycleMixin._api_kwargs_have_image_parts(kwargs))

    def test_anthropic_converter_passes_unhashable_types_through(self) -> None:
        for type_value in _UNHASHABLE_TYPES:
            with self.subTest(type_value=type_value):
                block = _convert_content_part_to_anthropic(
                    {"type": type_value, "text": "hi"}
                )
                self.assertIsInstance(block, dict)

    def test_shrink_pass_ignores_unhashable_types(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": ["string", "null"], "image_url": {"url": "data:x"}}],
            }
        ]
        self.assertFalse(try_shrink_image_parts_in_messages(messages))


if __name__ == "__main__":
    unittest.main()
