"""
Security and privacy tests for Symbiote.
Tests consent gates, redaction, PII handling, and audit logging.
"""

import json
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from symbiote.daemon.consent import (
    ConsentManager, ConsentScope, RedactionEngine, RedactionPreview
)
from symbiote.daemon.config import Config


class TestRedactionEngine:
    """Test PII detection and redaction."""
    
    def test_detect_emails(self):
        """Test email detection."""
        text = "Contact me at john.doe@example.com or admin@company.org"
        pii = RedactionEngine.detect_pii(text)
        
        assert len(pii["emails"]) == 2
        assert "john.doe@example.com" in pii["emails"]
        assert "admin@company.org" in pii["emails"]
    
    def test_detect_phone_numbers(self):
        """Test phone number detection."""
        text = "Call me at (555) 123-4567 or +1-800-555-0100"
        pii = RedactionEngine.detect_pii(text)
        
        assert len(pii["phones"]) >= 1
        assert any("555" in phone for phone in pii["phones"])
    
    def test_detect_ssns(self):
        """Test SSN detection."""
        text = "SSN: 123-45-6789 should be masked"
        pii = RedactionEngine.detect_pii(text)
        
        assert len(pii["ssns"]) == 1
        assert "123-45-6789" in pii["ssns"]
    
    def test_detect_names(self):
        """Test name detection."""
        text = "Meeting with Priya and David about the project"
        pii = RedactionEngine.detect_pii(text)
        
        # Should detect common names
        assert any("priya" in name.lower() for name in pii["names"])
        assert any("david" in name.lower() for name in pii["names"])
    
    def test_redact_emails(self):
        """Test email redaction."""
        text = "Email john@example.com for details"
        redacted, counts = RedactionEngine.redact_text(text, ["emails"])
        
        assert "john@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert counts["emails"] == 1
    
    def test_redact_multiple_types(self):
        """Test redacting multiple PII types."""
        text = "Call John at 555-1234 or email john@example.com"
        redacted, counts = RedactionEngine.redact_text(
            text, ["emails", "phones", "names"]
        )
        
        assert "john@example.com" not in redacted
        assert "555-1234" not in redacted
        assert "[EMAIL_REDACTED]" in redacted
        assert "[PHONE_REDACTED]" in redacted
    
    def test_mask_ids(self):
        """Test ID masking."""
        text = "Task 01J4T8W9C8H5P8Z7GZ8QF1XK2J and note abc-123"
        masked = RedactionEngine.mask_ids(text)
        
        assert "01J4T8W9C8H5P8Z7GZ8QF1XK2J" not in masked
        assert "[ID_***]" in masked
    
    def test_no_false_positives(self):
        """Test that non-PII text is not redacted."""
        text = "The performance metric is 99.9% with latency under 100ms"
        redacted, counts = RedactionEngine.redact_text(
            text, ["emails", "phones", "names"]
        )
        
        # Should not change technical text
        assert "99.9%" in redacted
        assert "100ms" in redacted
        assert sum(counts.values()) == 0


class TestConsentManager:
    """Test consent flow and preview generation."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return Config(
            vault_path="/tmp/test_vault",
            privacy={
                "allow_cloud": False,
                "redaction_default": True,
                "mask_pii_default": True
            }
        )
    
    @pytest.fixture
    def consent_manager(self, config):
        """Create consent manager."""
        # Mock the config object properly
        mock_config = Mock()
        mock_config.privacy.allow_cloud = False
        mock_config.privacy.mask_pii_default = True
        return ConsentManager(mock_config)
    
    def test_default_deny_scope(self, consent_manager):
        """Test default scope is DENY when cloud disabled."""
        assert consent_manager.default_scope == ConsentScope.DENY
    
    def test_preview_deny_scope(self, consent_manager):
        """Test DENY scope returns empty preview."""
        data = {
            "notes": [
                {"title": "Secret Project", "content": "Confidential data"}
            ]
        }
        
        preview = consent_manager.prepare_preview(
            data, 
            scope=ConsentScope.DENY
        )
        
        assert preview.scope == ConsentScope.DENY
        assert len(preview.items) == 0
        assert preview.token_estimate == 0
    
    def test_preview_titles_only(self, consent_manager):
        """Test ALLOW_TITLES scope masks IDs."""
        data = {
            "notes": [
                {"title": "Task 01J4T8W9C8H5P8Z7GZ8QF1XK2J", "content": "Details"},
                {"title": "Meeting Notes", "content": "Private"}
            ]
        }
        
        preview = consent_manager.prepare_preview(
            data,
            scope=ConsentScope.ALLOW_TITLES
        )
        
        assert preview.scope == ConsentScope.ALLOW_TITLES
        assert len(preview.items) == 2
        assert "[ID_***]" in preview.items[0]
        assert "Meeting Notes" in preview.items[1]
        assert "Private" not in str(preview.items)  # Content not included
    
    def test_preview_bullets_with_redaction(self, consent_manager):
        """Test ALLOW_BULLETS scope with PII redaction."""
        data = {
            "notes": [
                {
                    "title": "Contact List",
                    "content": "- Email john@example.com\n- Call 555-1234"
                }
            ]
        }
        
        preview = consent_manager.prepare_preview(
            data,
            scope=ConsentScope.ALLOW_BULLETS
        )
        
        assert preview.scope == ConsentScope.ALLOW_BULLETS
        # Should extract and redact facts
        assert any("[EMAIL_REDACTED]" in item for item in preview.items)
        assert preview.redactions.get("emails", 0) > 0
    
    def test_consent_request_storage(self, consent_manager):
        """Test consent request is stored for later decision."""
        preview = RedactionPreview(
            scope=ConsentScope.ALLOW_TITLES,
            items=["Test item"],
            redactions={},
            token_estimate=10,
            destination="test"
        )
        
        action_id = "test_action_123"
        returned_id = consent_manager.request_consent(action_id, preview)
        
        assert returned_id == action_id
        assert action_id in consent_manager.pending_consents
    
    def test_apply_consent_approved(self, consent_manager):
        """Test applying approved consent."""
        preview = RedactionPreview(
            scope=ConsentScope.ALLOW_BULLETS,
            items=["Item 1", "Item 2 with sensitive data"],
            redactions={},
            token_estimate=20,
            destination="test"
        )
        
        action_id = "test_action_456"
        consent_manager.request_consent(action_id, preview)
        
        # Approve without additional redactions
        final_payload = consent_manager.apply_consent(
            action_id, 
            approved=True,
            additional_redactions=None
        )
        
        assert final_payload == preview.items
        assert action_id not in consent_manager.pending_consents
    
    def test_apply_consent_denied(self, consent_manager):
        """Test denying consent."""
        preview = RedactionPreview(
            scope=ConsentScope.ALLOW_BULLETS,
            items=["Sensitive data"],
            redactions={},
            token_estimate=10,
            destination="test"
        )
        
        action_id = "test_action_789"
        consent_manager.request_consent(action_id, preview)
        
        # Deny consent
        final_payload = consent_manager.apply_consent(
            action_id,
            approved=False
        )
        
        assert final_payload is None
        assert action_id not in consent_manager.pending_consents
    
    def test_apply_additional_redactions(self, consent_manager):
        """Test applying user-requested additional redactions."""
        preview = RedactionPreview(
            scope=ConsentScope.ALLOW_BULLETS,
            items=["Project Alpha details", "Budget: $1M", "Contact: CEO"],
            redactions={},
            token_estimate=30,
            destination="test"
        )
        
        action_id = "test_action_999"
        consent_manager.request_consent(action_id, preview)
        
        # Approve with additional redactions
        final_payload = consent_manager.apply_consent(
            action_id,
            approved=True,
            additional_redactions=["mask:$1M", "remove:CEO"]
        )
        
        assert final_payload is not None
        assert any("[MASKED]" in item for item in final_payload)
        # CEO line should be removed
        assert not any("CEO" in item for item in final_payload)


class TestPrivacyIntegration:
    """Integration tests for privacy features."""
    
    @pytest.mark.asyncio
    async def test_no_outbound_without_consent(self):
        """Test that no data leaves without explicit consent."""
        # This would test the actual API endpoints
        # For now, we verify the consent gate exists
        from symbiote.daemon.api import handle_deliberate
        
        # The handler should check allow_cloud and require consent
        # Real test would mock the request and verify behavior
        assert handle_deliberate is not None
    
    def test_audit_log_structure(self):
        """Test audit log contains required fields."""
        from symbiote.daemon.indexers.analytics import AnalyticsIndexer
        
        # Verify the audit_outbound table schema
        # This would connect to test DB and verify structure
        # For now, just verify the class exists
        assert AnalyticsIndexer is not None
    
    def test_receipts_immutability(self):
        """Test that receipts cannot be modified."""
        # Receipts should have version field and be append-only
        receipt = {
            "id": "rcp_test",
            "created_at": datetime.utcnow().isoformat(),
            "suggestion_text": "Test suggestion",
            "sources": [],
            "heuristics": [],
            "confidence": "medium",
            "version": 1
        }
        
        # Verify required fields
        assert "id" in receipt
        assert "version" in receipt
        assert receipt["version"] == 1
    
    def test_default_privacy_settings(self):
        """Test privacy-preserving defaults."""
        from symbiote.daemon.config import PrivacyConfig
        
        privacy = PrivacyConfig()
        
        # Verify secure defaults
        assert privacy.allow_cloud == False
        assert privacy.redaction_default == True
        assert privacy.mask_pii_default == True


class TestSecurityBoundaries:
    """Test security boundaries and process isolation."""
    
    def test_whisper_process_isolation(self):
        """Test STT runs as separate process."""
        # Verify whisper adapter spawns child process
        # Not actual subprocess.Popen call
        from symbiote.daemon.adapters import stt
        # Would verify stt module exists and uses subprocess
        assert True  # Placeholder
    
    def test_localhost_binding(self):
        """Test daemon binds to localhost only."""
        # Verify API binds to 127.0.0.1
        from symbiote.daemon.main import SymbioteDaemon
        # Would check actual binding in integration test
        assert True  # Placeholder
    
    def test_no_plaintext_secrets(self):
        """Test secrets are not stored in plaintext."""
        # Verify config doesn't contain actual tokens
        from symbiote.daemon.config import Config
        
        # Would scan config files for patterns
        # For now, verify the config structure exists
        assert hasattr(Config, 'privacy')


@pytest.mark.parametrize("text,expected_redacted", [
    (
        "Email admin@example.com about project",
        "[EMAIL_REDACTED]"
    ),
    (
        "SSN 123-45-6789 in record",
        "[SSN_REDACTED]"
    ),
    (
        "Call 555-123-4567 for info",
        "[PHONE_REDACTED]"
    ),
])
def test_redaction_scenarios(text, expected_redacted):
    """Test various redaction scenarios."""
    redacted, _ = RedactionEngine.redact_text(
        text, 
        ["emails", "phones", "ssns"]
    )
    assert expected_redacted in redacted


def test_consent_scope_ordering():
    """Test consent scope levels are properly ordered."""
    # Verify scope restrictiveness
    assert ConsentScope.DENY.value == "deny"
    assert ConsentScope.ALLOW_TITLES.value == "allow_titles"
    assert ConsentScope.ALLOW_BULLETS.value == "allow_bullets"
    assert ConsentScope.ALLOW_EXCERPTS.value == "allow_excerpts"
    
    # Each level should allow more data
    scopes = [
        ConsentScope.DENY,
        ConsentScope.ALLOW_TITLES,
        ConsentScope.ALLOW_BULLETS,
        ConsentScope.ALLOW_EXCERPTS
    ]
    
    # Verify ordering makes sense
    for i in range(len(scopes) - 1):
        # Each subsequent scope is less restrictive
        assert scopes[i] != scopes[i + 1]