"""Consent and redaction pipeline for privacy protection."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class ConsentScope(Enum):
    """Levels of consent for outbound data."""
    DENY = "deny"
    ALLOW_TITLES = "allow_titles"
    ALLOW_BULLETS = "allow_bullets"
    ALLOW_EXCERPTS = "allow_excerpts"


@dataclass
class RedactionPreview:
    """Preview of what data would be sent externally."""
    scope: ConsentScope
    items: List[str]  # Exact lines that would be sent
    redactions: Dict[str, int]  # Type -> count of redactions
    token_estimate: int
    destination: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "items": self.items,
            "redactions": self.redactions,
            "token_estimate": self.token_estimate,
            "destination": self.destination
        }


class RedactionEngine:
    """Handles PII detection and redaction."""
    
    # Simplified PII patterns (production would use more sophisticated detection)
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    
    # Common names to mask (simplified)
    COMMON_NAMES = {
        "john", "jane", "bob", "alice", "priya", "kumar", "smith", "jones",
        "david", "michael", "sarah", "emily", "james", "mary", "patricia"
    }
    
    @staticmethod
    def detect_pii(text: str) -> Dict[str, List[str]]:
        """Detect potential PII in text."""
        pii_found = {
            "emails": [],
            "phones": [],
            "ssns": [],
            "names": []
        }
        
        # Find emails
        pii_found["emails"] = RedactionEngine.EMAIL_PATTERN.findall(text)
        
        # Find phone numbers
        pii_found["phones"] = RedactionEngine.PHONE_PATTERN.findall(text)
        
        # Find SSNs
        pii_found["ssns"] = RedactionEngine.SSN_PATTERN.findall(text)
        
        # Find potential names (simplified)
        words = text.lower().split()
        for word in words:
            clean_word = word.strip(".,!?;:'\"")
            if clean_word in RedactionEngine.COMMON_NAMES:
                pii_found["names"].append(word)
        
        return pii_found
    
    @staticmethod
    def redact_text(text: str, redaction_types: List[str]) -> Tuple[str, Dict[str, int]]:
        """
        Apply redactions to text.
        Returns (redacted_text, redaction_counts).
        """
        redacted = text
        counts = {}
        
        if "emails" in redaction_types:
            emails = RedactionEngine.EMAIL_PATTERN.findall(redacted)
            for email in emails:
                redacted = redacted.replace(email, "[EMAIL_REDACTED]")
            counts["emails"] = len(emails)
        
        if "phones" in redaction_types:
            phones = RedactionEngine.PHONE_PATTERN.findall(redacted)
            for phone in phones:
                redacted = redacted.replace(phone, "[PHONE_REDACTED]")
            counts["phones"] = len(phones)
        
        if "ssns" in redaction_types:
            ssns = RedactionEngine.SSN_PATTERN.findall(redacted)
            for ssn in ssns:
                redacted = redacted.replace(ssn, "[SSN_REDACTED]")
            counts["ssns"] = len(ssns)
        
        if "names" in redaction_types:
            name_count = 0
            words = redacted.split()
            for i, word in enumerate(words):
                clean_word = word.lower().strip(".,!?;:'\"")
                if clean_word in RedactionEngine.COMMON_NAMES:
                    # Preserve punctuation
                    prefix = word[:len(word) - len(word.lstrip(".,!?;:'\""))]
                    suffix = word[len(word.rstrip(".,!?;:'\"")):]
                    words[i] = f"{prefix}[NAME]{suffix}"
                    name_count += 1
            redacted = " ".join(words)
            counts["names"] = name_count
        
        return redacted, counts
    
    @staticmethod
    def mask_ids(text: str) -> str:
        """Mask IDs to prevent correlation."""
        # Mask ULID-style IDs
        ulid_pattern = re.compile(r'\b[0-9A-Z]{26}\b')
        masked = ulid_pattern.sub("[ID_***]", text)
        
        # Mask UUID-style IDs
        uuid_pattern = re.compile(
            r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b',
            re.IGNORECASE
        )
        masked = uuid_pattern.sub("[UUID_***]", masked)
        
        return masked


class ConsentManager:
    """Manages consent flow and audit logging."""
    
    def __init__(self, config):
        self.config = config
        self.default_scope = ConsentScope.DENY if not config.privacy.allow_cloud else ConsentScope.ALLOW_TITLES
        self.pending_consents = {}  # action_id -> preview
    
    def prepare_preview(
        self,
        data: Dict[str, Any],
        destination: str = "cloud:generic",
        scope: Optional[ConsentScope] = None
    ) -> RedactionPreview:
        """
        Prepare a preview of what would be sent.
        """
        if scope is None:
            scope = self.default_scope
        
        items = []
        redaction_counts = {}
        
        if scope == ConsentScope.DENY:
            return RedactionPreview(
                scope=scope,
                items=[],
                redactions={},
                token_estimate=0,
                destination=destination
            )
        
        # Extract data based on scope
        if scope == ConsentScope.ALLOW_TITLES:
            # Only titles and masked IDs
            if "notes" in data:
                for note in data["notes"]:
                    title = note.get("title", "Untitled")
                    masked_title = RedactionEngine.mask_ids(title)
                    items.append(f"Note: {masked_title}")
            
            if "tasks" in data:
                for task in data["tasks"]:
                    title = task.get("title", "Untitled")
                    masked_title = RedactionEngine.mask_ids(title)
                    items.append(f"Task: {masked_title}")
        
        elif scope == ConsentScope.ALLOW_BULLETS:
            # Titles + bulletized facts
            if "notes" in data:
                for note in data["notes"]:
                    title = note.get("title", "Untitled")
                    items.append(f"Note: {title}")
                    
                    # Extract key facts as bullets
                    content = note.get("content", "")
                    facts = self._extract_facts(content)
                    for fact in facts[:3]:  # Limit bullets
                        redacted_fact, counts = RedactionEngine.redact_text(
                            fact,
                            ["emails", "phones", "ssns", "names"] if self.config.privacy.mask_pii_default else []
                        )
                        items.append(f"  • {redacted_fact}")
                        for k, v in counts.items():
                            redaction_counts[k] = redaction_counts.get(k, 0) + v
        
        elif scope == ConsentScope.ALLOW_EXCERPTS:
            # Include small excerpts
            if "notes" in data:
                for note in data["notes"]:
                    title = note.get("title", "Untitled")
                    items.append(f"Note: {title}")
                    
                    content = note.get("content", "")
                    excerpt = content[:200] + "..." if len(content) > 200 else content
                    redacted_excerpt, counts = RedactionEngine.redact_text(
                        excerpt,
                        ["emails", "phones", "ssns"] if self.config.privacy.mask_pii_default else []
                    )
                    items.append(f"  {redacted_excerpt}")
                    for k, v in counts.items():
                        redaction_counts[k] = redaction_counts.get(k, 0) + v
        
        # Estimate tokens (rough: 1 token per 4 chars)
        total_chars = sum(len(item) for item in items)
        token_estimate = total_chars // 4
        
        return RedactionPreview(
            scope=scope,
            items=items,
            redactions=redaction_counts,
            token_estimate=token_estimate,
            destination=destination
        )
    
    def _extract_facts(self, content: str, max_facts: int = 5) -> List[str]:
        """Extract key facts from content."""
        facts = []
        
        # Simple extraction: look for lines starting with -, *, or numbered
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line and (
                line.startswith('- ') or
                line.startswith('* ') or
                line.startswith('• ') or
                (len(line) > 2 and line[0].isdigit() and line[1] in '.)')
            ):
                # Clean up bullet markers
                fact = re.sub(r'^[-*•]\s*', '', line)
                fact = re.sub(r'^\d+[.)]\s*', '', fact)
                if len(fact) > 10:  # Minimum fact length
                    facts.append(fact)
                    if len(facts) >= max_facts:
                        break
        
        return facts
    
    def request_consent(
        self,
        action_id: str,
        preview: RedactionPreview
    ) -> str:
        """
        Store preview for consent decision.
        Returns action_id for tracking.
        """
        self.pending_consents[action_id] = preview
        logger.info(f"Consent requested for action {action_id}: {len(preview.items)} items")
        return action_id
    
    def apply_consent(
        self,
        action_id: str,
        approved: bool,
        additional_redactions: Optional[List[str]] = None
    ) -> Optional[List[str]]:
        """
        Apply user's consent decision.
        Returns final payload if approved, None if denied.
        """
        if action_id not in self.pending_consents:
            logger.error(f"Unknown action_id: {action_id}")
            return None
        
        preview = self.pending_consents.pop(action_id)
        
        if not approved:
            logger.info(f"Consent denied for action {action_id}")
            return None
        
        # Apply any additional redactions requested by user
        final_items = preview.items
        if additional_redactions:
            final_items = []
            for item in preview.items:
                redacted_item = item
                for redaction in additional_redactions:
                    if redaction.startswith("remove:"):
                        pattern = redaction[7:]
                        if pattern in redacted_item:
                            continue  # Skip this item entirely
                    elif redaction.startswith("mask:"):
                        pattern = redaction[5:]
                        redacted_item = redacted_item.replace(pattern, "[MASKED]")
                final_items.append(redacted_item)
        
        logger.info(f"Consent approved for action {action_id}: {len(final_items)} items")
        return final_items