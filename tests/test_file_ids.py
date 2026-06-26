"""
Test File ID Tracking - Lossless Guarantee

Tests for file_ids.py module to verify ZERO information loss.
"""

import pytest

from memoryforge.lcm import Message, file_ids


class TestFileIDExtraction:
    """Test file ID extraction from text"""

    def test_extract_single_file_id(self):
        text = "Check file_a1b2c3d4 for details"
        result = file_ids.extract_file_ids(text)
        assert result == ["file_a1b2c3d4"]

    def test_extract_multiple_file_ids(self):
        text = "See file_a1b2c3d4 and file_deadbeef for context"
        result = file_ids.extract_file_ids(text)
        assert result == ["file_a1b2c3d4", "file_deadbeef"]

    def test_deduplication(self):
        text = "file_abc123 mentioned twice: file_abc123"
        result = file_ids.extract_file_ids(text)
        assert result == ["file_abc123"]
        assert len(result) == 1

    def test_no_file_ids(self):
        text = "This has no file references"
        result = file_ids.extract_file_ids(text)
        assert result == []

    def test_various_lengths(self):
        # 6 chars min
        assert file_ids.extract_file_ids("file_a1b2c3") == ["file_a1b2c3"]

        # 16 chars
        assert file_ids.extract_file_ids("file_a1b2c3d4e5f6a7b8") == ["file_a1b2c3d4e5f6a7b8"]

        # 32 chars max
        long_id = "file_" + "a" * 32
        assert file_ids.extract_file_ids(long_id) == [long_id]

    def test_invalid_formats(self):
        # Too short
        assert file_ids.extract_file_ids("file_abc") == []

        # No prefix
        assert file_ids.extract_file_ids("notfile_a1b2c3d4") == []

        # Non-hex characters
        assert file_ids.extract_file_ids("file_xyz12345") == []


class TestFileIDFooter:
    """Test footer append/strip operations"""

    def test_append_footer(self):
        text = "Summary text"
        result = file_ids.append_file_ids_footer(text, ["file_abc123"])
        assert "[LCM File IDs: file_abc123]" in result

    def test_append_multiple_ids(self):
        text = "Summary"
        result = file_ids.append_file_ids_footer(text, ["file_abc", "file_def"])
        assert "[LCM File IDs: file_abc, file_def]" in result

    def test_idempotent_append(self):
        """Footer append should be idempotent - no duplication"""
        text = "Summary\n\n[LCM File IDs: file_old]"
        result = file_ids.append_file_ids_footer(text, ["file_new"])

        # Should have only ONE footer with new IDs
        assert result.count("[LCM File IDs:") == 1
        assert "file_new" in result
        assert "file_old" not in result

    def test_empty_file_ids_list(self):
        text = "Summary"
        result = file_ids.append_file_ids_footer(text, [])
        assert result == text  # No footer added

    def test_strip_footer(self):
        text = "Summary\n\n[LCM File IDs: file_abc, file_def]"
        result = file_ids.strip_file_ids_footer(text)
        assert result == "Summary"
        assert "[LCM File IDs:" not in result


class TestMessageExtraction:
    """Test extraction from message lists"""

    def test_extract_from_messages(self):
        messages = [
            Message(role="user", content="Check file_abc123"),
            Message(role="assistant", content="Analyzed file_def456"),
        ]
        result = file_ids.extract_file_ids_from_messages(messages)
        assert result == ["file_abc123", "file_def456"]

    def test_deduplication_across_messages(self):
        messages = [
            Message(role="user", content="file_abc123"),
            Message(role="assistant", content="file_abc123 and file_def456"),
        ]
        result = file_ids.extract_file_ids_from_messages(messages)
        # Should deduplicate but preserve order
        assert result == ["file_abc123", "file_def456"]

    def test_truncate_long_messages(self):
        """Very long messages should be truncated for safety"""
        huge_content = "file_abc123 " + ("x" * 200_000)  # 200K chars
        messages = [Message(role="user", content=huge_content)]

        # Should not crash, should still extract file ID
        result = file_ids.extract_file_ids_from_messages(messages, max_chars_per_message=100_000)
        assert "file_abc123" in result


class TestValidation:
    """Test file ID format validation"""

    def test_valid_formats(self):
        assert file_ids.validate_file_id_format("file_a1b2c3d4") is True
        assert file_ids.validate_file_id_format("file_deadbeef12345678") is True

    def test_invalid_formats(self):
        assert file_ids.validate_file_id_format("file_abc") is False  # Too short
        assert file_ids.validate_file_id_format("notfile_a1b2c3d4") is False  # No prefix
        assert file_ids.validate_file_id_format("file_xyz12345") is False  # Non-hex


class TestLosslessGuarantee:
    """Test the core lossless guarantee"""

    def test_verify_lossless_success(self):
        """All input file IDs present in output → lossless"""
        messages = [
            Message(role="user", content="Check file_abc123 and file_def456"),
        ]
        output = "Summary text\n\n[LCM File IDs: file_abc123, file_def456]"

        assert file_ids.verify_lossless(messages, output) is True

    def test_verify_lossless_failure(self):
        """Missing file ID in output → NOT lossless"""
        messages = [
            Message(role="user", content="Check file_abc123 and file_def456"),
        ]
        output = "Summary text\n\n[LCM File IDs: file_abc123]"  # Missing file_def456

        assert file_ids.verify_lossless(messages, output) is False

    def test_verify_lossless_extra_ids_ok(self):
        """Output can have MORE file IDs (from previous rounds)"""
        messages = [
            Message(role="user", content="Check file_abc123"),
        ]
        output = "Summary\n\n[LCM File IDs: file_abc123, file_old_from_prev_round]"

        # This is OK - we preserve old IDs too
        assert file_ids.verify_lossless(messages, output) is True


class TestRealWorldScenarios:
    """Test realistic usage patterns"""

    def test_compaction_round_preserves_ids(self):
        """Simulate L3 deterministic compaction"""
        messages = [
            Message(role="user", content="Analyze file_abcd2024"),
            Message(role="assistant", content="Results in file_abc123def"),
        ]

        # Extract IDs
        all_ids = file_ids.extract_file_ids_from_messages(messages)
        assert len(all_ids) == 2

        # Build summary (simulated L3)
        summary = "[L3 DETERMINISTIC]\nAnalysis completed."
        summary = file_ids.append_file_ids_footer(summary, all_ids)

        # Verify lossless
        assert file_ids.verify_lossless(messages, summary)

    def test_multi_round_condensation(self):
        """Simulate condensing multiple summaries"""
        # Round 1 summaries
        summary1 = "Summary A\n\n[LCM File IDs: file_aaaaaa, file_bbbbbb]"
        summary2 = "Summary B\n\n[LCM File IDs: file_bbbbbb, file_cccccc]"

        # Extract from both
        ids1 = file_ids.extract_file_ids(summary1)
        ids2 = file_ids.extract_file_ids(summary2)

        # Merge for condensation
        all_ids = list(dict.fromkeys(ids1 + ids2))  # Dedupe preserving order
        assert set(all_ids) == {"file_aaaaaa", "file_bbbbbb", "file_cccccc"}

        # Build condensed summary
        condensed = "Condensed AB"
        condensed = file_ids.append_file_ids_footer(condensed, all_ids)

        # Verify all IDs preserved
        assert all(fid in condensed for fid in ["file_aaaaaa", "file_bbbbbb", "file_cccccc"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
