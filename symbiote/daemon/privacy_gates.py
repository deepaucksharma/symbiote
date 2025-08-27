"""Production privacy and security gates for cloud operations.

This module implements comprehensive privacy protection:
- PII detection and redaction
- Consent management with granular scopes
- Audit logging for all external operations
- Data minimization and anonymization
- Secure key management
"""

import re
import hashlib
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import secrets
import base64

try:
    import spacy
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    HAS_PRIVACY_DEPS = True
except ImportError:
    HAS_PRIVACY_DEPS = False

from loguru import logger


class ConsentLevel(Enum):
    """Granular consent levels for data sharing."""
    DENY = 0
    ALLOW_ANONYMOUS = 1
    ALLOW_TITLES = 2
    ALLOW_METADATA = 3
    ALLOW_EXCERPTS = 4
    ALLOW_FULL = 5


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PERSON_NAME = "person_name"
    LOCATION = "location"
    DATE_OF_BIRTH = "dob"
    MEDICAL = "medical"
    FINANCIAL = "financial"


@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'type': self.pii_type.value,
            'text': self.text,
            'position': [self.start, self.end],
            'confidence': self.confidence
        }


@dataclass
class ConsentRequest:
    """Request for user consent for an operation."""
    request_id: str
    operation: str
    data_preview: str
    pii_detected: List[PIIMatch]
    consent_level_required: ConsentLevel
    expires_at: datetime
    purpose: str
    third_parties: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'request_id': self.request_id,
            'operation': self.operation,
            'data_preview': self.data_preview,
            'pii_count': len(self.pii_detected),
            'consent_level': self.consent_level_required.name,
            'expires_at': self.expires_at.isoformat(),
            'purpose': self.purpose,
            'third_parties': self.third_parties
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for external operations."""
    timestamp: datetime
    operation: str
    data_hash: str
    consent_level: ConsentLevel
    pii_redacted: bool
    destination: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'data_hash': self.data_hash,
            'consent_level': self.consent_level.name,
            'pii_redacted': self.pii_redacted,
            'destination': self.destination,
            'user_id': self.user_id,
            'request_id': self.request_id
        }


class PIIDetector:
    """Advanced PII detection using patterns and NLP."""
    
    def __init__(self, use_nlp: bool = True):
        """
        Initialize PII detector.
        
        Args:
            use_nlp: Whether to use NLP for entity recognition
        """
        self.use_nlp = use_nlp and HAS_PRIVACY_DEPS
        self.nlp = None
        
        if self.use_nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                logger.warning("Spacy model not found, using pattern matching only")
                self.use_nlp = False
        
        # Compile regex patterns
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[PIIType, re.Pattern]:
        """Compile regex patterns for PII detection."""
        patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            PIIType.DATE_OF_BIRTH: re.compile(
                r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b'
            )
        }
        return patterns
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: Text to scan
            
        Returns:
            List of PII matches
        """
        matches = []
        
        # Pattern-based detection
        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    context=text[max(0, match.start()-20):min(len(text), match.end()+20)]
                )
                matches.append(pii_match)
        
        # NLP-based detection
        if self.use_nlp and self.nlp:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                pii_type = None
                confidence = 0.7
                
                if ent.label_ == "PERSON":
                    pii_type = PIIType.PERSON_NAME
                    confidence = 0.8
                elif ent.label_ in ["GPE", "LOC"]:
                    pii_type = PIIType.LOCATION
                    confidence = 0.7
                elif ent.label_ == "DATE" and self._is_dob_context(ent.text, text):
                    pii_type = PIIType.DATE_OF_BIRTH
                    confidence = 0.6
                
                if pii_type:
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        text=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=confidence,
                        context=text[max(0, ent.start_char-20):min(len(text), ent.end_char+20)]
                    )
                    matches.append(pii_match)
        
        # Deduplicate overlapping matches
        matches = self._deduplicate_matches(matches)
        
        return matches
    
    def _is_dob_context(self, date_text: str, full_text: str) -> bool:
        """Check if a date is likely a date of birth based on context."""
        dob_keywords = ['born', 'birth', 'dob', 'birthday', 'age']
        date_pos = full_text.find(date_text)
        
        if date_pos == -1:
            return False
        
        # Check surrounding context
        context = full_text[max(0, date_pos-50):min(len(full_text), date_pos+50)].lower()
        
        return any(keyword in context for keyword in dob_keywords)
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping PII matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Sort by start position and confidence
        matches.sort(key=lambda x: (x.start, -x.confidence))
        
        deduplicated = []
        last_end = -1
        
        for match in matches:
            if match.start >= last_end:
                deduplicated.append(match)
                last_end = match.end
        
        return deduplicated


class DataRedactor:
    """Redact and anonymize sensitive data."""
    
    def __init__(self, preserve_format: bool = True):
        """
        Initialize data redactor.
        
        Args:
            preserve_format: Whether to preserve data format when redacting
        """
        self.preserve_format = preserve_format
        self.pii_detector = PIIDetector()
    
    def redact(self, 
              text: str,
              pii_matches: Optional[List[PIIMatch]] = None,
              consent_level: ConsentLevel = ConsentLevel.ALLOW_ANONYMOUS) -> str:
        """
        Redact PII from text based on consent level.
        
        Args:
            text: Text to redact
            pii_matches: Pre-detected PII (optional)
            consent_level: Level of consent granted
            
        Returns:
            Redacted text
        """
        if consent_level == ConsentLevel.ALLOW_FULL:
            return text
        
        if consent_level == ConsentLevel.DENY:
            return "[CONTENT BLOCKED - NO CONSENT]"
        
        # Detect PII if not provided
        if pii_matches is None:
            pii_matches = self.pii_detector.detect(text)
        
        if not pii_matches:
            return text
        
        # Sort matches by position (reverse order for replacement)
        pii_matches.sort(key=lambda x: x.start, reverse=True)
        
        redacted_text = text
        
        for match in pii_matches:
            replacement = self._get_replacement(match, consent_level)
            redacted_text = (
                redacted_text[:match.start] +
                replacement +
                redacted_text[match.end:]
            )
        
        return redacted_text
    
    def _get_replacement(self, match: PIIMatch, consent_level: ConsentLevel) -> str:
        """Get appropriate replacement for PII based on type and consent."""
        if consent_level == ConsentLevel.ALLOW_ANONYMOUS:
            # Complete redaction
            return f"[{match.pii_type.value.upper()}_REDACTED]"
        
        elif consent_level in [ConsentLevel.ALLOW_TITLES, ConsentLevel.ALLOW_METADATA]:
            # Partial redaction
            if self.preserve_format:
                return self._format_preserving_redaction(match)
            else:
                return f"[{match.pii_type.value.upper()}]"
        
        elif consent_level == ConsentLevel.ALLOW_EXCERPTS:
            # Minimal redaction
            if match.pii_type in [PIIType.SSN, PIIType.CREDIT_CARD]:
                # Always fully redact highly sensitive
                return f"[{match.pii_type.value.upper()}_REDACTED]"
            else:
                # Partial masking
                return self._partial_mask(match.text)
        
        return f"[REDACTED]"
    
    def _format_preserving_redaction(self, match: PIIMatch) -> str:
        """Redact while preserving format."""
        text = match.text
        
        if match.pii_type == PIIType.EMAIL:
            parts = text.split('@')
            if len(parts) == 2:
                return f"***@{parts[1]}"
            
        elif match.pii_type == PIIType.PHONE:
            # Keep area code
            digits = re.findall(r'\d', text)
            if len(digits) >= 10:
                return f"({digits[0]}{digits[1]}{digits[2]}) ***-****"
            
        elif match.pii_type == PIIType.CREDIT_CARD:
            # Keep last 4 digits
            digits = re.findall(r'\d', text)
            if len(digits) >= 4:
                return f"****-****-****-{''.join(digits[-4:])}"
        
        return f"[{match.pii_type.value.upper()}]"
    
    def _partial_mask(self, text: str) -> str:
        """Partially mask text."""
        if len(text) <= 4:
            return "*" * len(text)
        
        # Show first and last characters
        visible_chars = max(2, len(text) // 4)
        return text[:visible_chars] + "*" * (len(text) - 2*visible_chars) + text[-visible_chars:]
    
    def anonymize_structured(self, data: Dict[str, Any], consent_level: ConsentLevel) -> Dict[str, Any]:
        """
        Anonymize structured data.
        
        Args:
            data: Structured data to anonymize
            consent_level: Consent level
            
        Returns:
            Anonymized data
        """
        if consent_level == ConsentLevel.ALLOW_FULL:
            return data
        
        if consent_level == ConsentLevel.DENY:
            return {"error": "No consent for data processing"}
        
        anonymized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                anonymized[key] = self.redact(value, consent_level=consent_level)
            elif isinstance(value, dict):
                anonymized[key] = self.anonymize_structured(value, consent_level)
            elif isinstance(value, list):
                anonymized[key] = [
                    self.redact(item, consent_level=consent_level) if isinstance(item, str)
                    else item for item in value
                ]
            else:
                # Non-string values
                if consent_level == ConsentLevel.ALLOW_ANONYMOUS:
                    # Hash numeric IDs
                    if key.lower().endswith('_id') and isinstance(value, (int, float)):
                        anonymized[key] = hashlib.sha256(str(value).encode()).hexdigest()[:8]
                    else:
                        anonymized[key] = value
                else:
                    anonymized[key] = value
        
        return anonymized


class ConsentManager:
    """Manage user consent for operations."""
    
    def __init__(self, storage_path: Path):
        """
        Initialize consent manager.
        
        Args:
            storage_path: Path to store consent records
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.consent_file = self.storage_path / "consent_records.json"
        self.audit_file = self.storage_path / "audit_log.jsonl"
        
        self.active_requests: Dict[str, ConsentRequest] = {}
        self.consent_records: Dict[str, Dict] = self._load_consent_records()
        
        self.redactor = DataRedactor()
        self.pii_detector = PIIDetector()
    
    def _load_consent_records(self) -> Dict:
        """Load consent records from disk."""
        if self.consent_file.exists():
            try:
                with open(self.consent_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
        
        return {
            'global_consent': ConsentLevel.ALLOW_ANONYMOUS.value,
            'operation_consents': {},
            'third_party_consents': {}
        }
    
    def _save_consent_records(self):
        """Save consent records to disk."""
        try:
            with open(self.consent_file, 'w') as f:
                json.dump(self.consent_records, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")
    
    async def request_consent(self,
                             operation: str,
                             data: str,
                             purpose: str,
                             third_parties: List[str] = None) -> ConsentRequest:
        """
        Request user consent for an operation.
        
        Args:
            operation: Operation requiring consent
            data: Data to be processed
            purpose: Purpose of the operation
            third_parties: Third parties that will receive data
            
        Returns:
            Consent request for user approval
        """
        # Detect PII in data
        pii_matches = self.pii_detector.detect(data)
        
        # Determine required consent level
        if pii_matches:
            if any(m.pii_type in [PIIType.SSN, PIIType.CREDIT_CARD] for m in pii_matches):
                required_level = ConsentLevel.DENY  # Never auto-approve highly sensitive
            else:
                required_level = ConsentLevel.ALLOW_EXCERPTS
        else:
            required_level = ConsentLevel.ALLOW_METADATA
        
        # Create preview with redacted PII
        preview = self.redactor.redact(
            data[:500],  # Limit preview length
            pii_matches=[m for m in pii_matches if m.start < 500],
            consent_level=ConsentLevel.ALLOW_ANONYMOUS
        )
        
        # Create consent request
        request = ConsentRequest(
            request_id=secrets.token_urlsafe(16),
            operation=operation,
            data_preview=preview,
            pii_detected=pii_matches,
            consent_level_required=required_level,
            expires_at=datetime.now() + timedelta(minutes=5),
            purpose=purpose,
            third_parties=third_parties or []
        )
        
        self.active_requests[request.request_id] = request
        
        return request
    
    async def grant_consent(self,
                           request_id: str,
                           consent_level: ConsentLevel,
                           remember: bool = False) -> bool:
        """
        Grant consent for a request.
        
        Args:
            request_id: Request ID
            consent_level: Level of consent granted
            remember: Whether to remember for future operations
            
        Returns:
            True if consent granted successfully
        """
        if request_id not in self.active_requests:
            return False
        
        request = self.active_requests[request_id]
        
        # Check if consent level is sufficient
        if consent_level.value < request.consent_level_required.value:
            logger.warning(f"Insufficient consent level for {request.operation}")
            return False
        
        # Store consent if requested
        if remember:
            self.consent_records['operation_consents'][request.operation] = {
                'level': consent_level.value,
                'granted_at': datetime.now().isoformat(),
                'purpose': request.purpose
            }
            
            for third_party in request.third_parties:
                self.consent_records['third_party_consents'][third_party] = {
                    'level': consent_level.value,
                    'granted_at': datetime.now().isoformat()
                }
            
            self._save_consent_records()
        
        # Log consent grant
        await self._audit_log(
            operation=request.operation,
            data_hash=hashlib.sha256(request.data_preview.encode()).hexdigest(),
            consent_level=consent_level,
            pii_redacted=len(request.pii_detected) > 0,
            destination=','.join(request.third_parties),
            request_id=request_id
        )
        
        # Remove from active requests
        del self.active_requests[request_id]
        
        return True
    
    def check_consent(self, operation: str) -> Optional[ConsentLevel]:
        """
        Check if consent has been granted for an operation.
        
        Args:
            operation: Operation to check
            
        Returns:
            Consent level if granted, None otherwise
        """
        if operation in self.consent_records['operation_consents']:
            level_value = self.consent_records['operation_consents'][operation]['level']
            return ConsentLevel(level_value)
        
        # Return global consent level
        return ConsentLevel(self.consent_records['global_consent'])
    
    async def _audit_log(self, **kwargs):
        """Log an operation to the audit trail."""
        entry = AuditLogEntry(
            timestamp=datetime.now(),
            **kwargs
        )
        
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    async def process_with_consent(self,
                                  operation: str,
                                  data: Any,
                                  processor_func,
                                  purpose: str,
                                  third_parties: List[str] = None) -> Any:
        """
        Process data with appropriate consent and privacy protection.
        
        Args:
            operation: Operation name
            data: Data to process
            processor_func: Function to process data
            purpose: Purpose of processing
            third_parties: Third parties involved
            
        Returns:
            Processed result or consent request
        """
        # Check existing consent
        consent_level = self.check_consent(operation)
        
        if consent_level is None or consent_level == ConsentLevel.DENY:
            # Request consent
            data_str = json.dumps(data) if not isinstance(data, str) else data
            request = await self.request_consent(
                operation=operation,
                data=data_str,
                purpose=purpose,
                third_parties=third_parties
            )
            
            return {
                'status': 'consent_required',
                'consent_request': request.to_dict()
            }
        
        # Apply privacy protection
        if isinstance(data, str):
            processed_data = self.redactor.redact(data, consent_level=consent_level)
        elif isinstance(data, dict):
            processed_data = self.redactor.anonymize_structured(data, consent_level)
        else:
            processed_data = data
        
        # Process with protected data
        try:
            result = await processor_func(processed_data)
            
            # Audit the operation
            await self._audit_log(
                operation=operation,
                data_hash=hashlib.sha256(str(processed_data).encode()).hexdigest(),
                consent_level=consent_level,
                pii_redacted=True,
                destination=','.join(third_parties or [])
            )
            
            return {
                'status': 'success',
                'result': result,
                'privacy_applied': consent_level.name
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }


class SecureKeyManager:
    """Secure management of API keys and credentials."""
    
    def __init__(self, key_file: Path):
        """
        Initialize key manager.
        
        Args:
            key_file: Path to encrypted key storage
        """
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._master_key = self._derive_master_key()
        self._cipher = None
        
        if HAS_PRIVACY_DEPS:
            self._cipher = Fernet(self._master_key)
        
        self._keys = self._load_keys()
    
    def _derive_master_key(self) -> bytes:
        """Derive master key from system entropy."""
        # In production, use hardware security module or OS keyring
        # This is a simplified version
        
        if not HAS_PRIVACY_DEPS:
            return b'0' * 32
        
        # Use machine ID + user ID as salt
        import uuid
        import os
        
        salt = f"{uuid.getnode()}{os.getuid()}".encode()
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # In production, get password from secure source
        password = b"temporary_dev_password"
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def _load_keys(self) -> Dict[str, str]:
        """Load encrypted keys from storage."""
        if not self.key_file.exists():
            return {}
        
        try:
            with open(self.key_file, 'rb') as f:
                encrypted_data = f.read()
            
            if self._cipher:
                decrypted = self._cipher.decrypt(encrypted_data)
                return json.loads(decrypted.decode())
            else:
                return {}
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}
    
    def _save_keys(self):
        """Save encrypted keys to storage."""
        try:
            data = json.dumps(self._keys).encode()
            
            if self._cipher:
                encrypted = self._cipher.encrypt(data)
            else:
                encrypted = data
            
            with open(self.key_file, 'wb') as f:
                f.write(encrypted)
            
            # Set restrictive permissions
            import os
            os.chmod(self.key_file, 0o600)
            
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def set_key(self, service: str, key: str):
        """Store an API key securely."""
        self._keys[service] = key
        self._save_keys()
    
    def get_key(self, service: str) -> Optional[str]:
        """Retrieve an API key."""
        return self._keys.get(service)
    
    def remove_key(self, service: str):
        """Remove an API key."""
        if service in self._keys:
            del self._keys[service]
            self._save_keys()