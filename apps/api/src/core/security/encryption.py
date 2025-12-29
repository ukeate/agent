"""
Encryption Service for Emotional Memory Management
Implements AES-256-GCM encryption for sensitive emotional data
"""

from src.core.utils.timezone_utils import utc_now
import os
import base64
import json
from typing import Tuple, Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import secrets
from datetime import datetime, timedelta
from ..exceptions import EncryptionException

from src.core.logging import get_logger
logger = get_logger(__name__)

class EncryptionService:
    """
    Service for encrypting and decrypting sensitive emotional memory data
    Uses AES-256-GCM for authenticated encryption
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption service with master key
        
        Args:
            master_key: Base64 encoded master key. If not provided, reads from environment
        """
        if master_key:
            self.master_key = base64.b64decode(master_key)
        else:
            # Read from environment variable
            key_env = os.environ.get('EMOTIONAL_MEMORY_MASTER_KEY')
            if not key_env:
                # Generate a new key if not exists (for development only)
                logger.warning("No master key found, generating new one. This should not happen in production!")
                self.master_key = secrets.token_bytes(32)
                logger.info(f"Generated master key: {base64.b64encode(self.master_key).decode()}")
            else:
                self.master_key = base64.b64decode(key_env)

        # Key cache for performance
        self._key_cache: Dict[str, Tuple[bytes, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)
    
    async def encrypt_data(
        self,
        plaintext: str,
        additional_data: Optional[bytes] = None
    ) -> Tuple[str, str]:
        """
        Encrypt sensitive data using AES-256-GCM
        
        Args:
            plaintext: Data to encrypt
            additional_data: Additional authenticated data (not encrypted but authenticated)
            
        Returns:
            Tuple of (encrypted_data_base64, key_id)
        """
        try:
            key_id = self._generate_key_id()
            data_key = await self._get_data_key(key_id)
            
            # Generate random nonce (96 bits for GCM)
            nonce = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(data_key),
                modes.GCM(nonce),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            # Add additional authenticated data if provided
            if additional_data:
                encryptor.authenticate_additional_data(additional_data)
            
            # Encrypt the data
            plaintext_bytes = plaintext.encode('utf-8')
            ciphertext = encryptor.update(plaintext_bytes) + encryptor.finalize()
            
            # Combine nonce, ciphertext, and auth tag
            encrypted_data = nonce + ciphertext + encryptor.tag
            
            # Encode to base64 for storage
            encrypted_base64 = base64.b64encode(encrypted_data).decode('utf-8')
            
            logger.debug(f"Encrypted data with key_id: {key_id}")
            
            return encrypted_base64, key_id
            
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise EncryptionException(f"Failed to encrypt data: {str(e)}")
    
    async def decrypt_data(
        self,
        encrypted_base64: str,
        key_id: str,
        additional_data: Optional[bytes] = None
    ) -> str:
        """
        Decrypt data encrypted with AES-256-GCM
        
        Args:
            encrypted_base64: Base64 encoded encrypted data
            key_id: ID of the encryption key used
            additional_data: Additional authenticated data used during encryption
            
        Returns:
            Decrypted plaintext
        """
        try:
            # Get data key from cache or derive it
            data_key = await self._get_data_key(key_id)
            
            # Decode from base64
            encrypted_data = base64.b64decode(encrypted_base64)
            
            # Extract components
            nonce = encrypted_data[:12]
            tag = encrypted_data[-16:]
            ciphertext = encrypted_data[12:-16]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(data_key),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            # Add additional authenticated data if provided
            if additional_data:
                decryptor.authenticate_additional_data(additional_data)
            
            # Decrypt the data
            plaintext_bytes = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Decode to string
            plaintext = plaintext_bytes.decode('utf-8')
            
            logger.debug(f"Decrypted data with key_id: {key_id}")
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise EncryptionException(f"Failed to decrypt data: {str(e)}")
    
    async def encrypt_json(
        self,
        data: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Encrypt JSON data
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Tuple of (encrypted_data_base64, key_id)
        """
        json_str = json.dumps(data, default=str)
        return await self.encrypt_data(json_str)
    
    async def decrypt_json(
        self,
        encrypted_base64: str,
        key_id: str
    ) -> Dict[str, Any]:
        """
        Decrypt JSON data
        
        Args:
            encrypted_base64: Base64 encoded encrypted data
            key_id: ID of the encryption key used
            
        Returns:
            Decrypted dictionary
        """
        json_str = await self.decrypt_data(encrypted_base64, key_id)
        return json.loads(json_str)
    
    def _generate_key_id(self) -> str:
        """
        Generate unique key identifier
        
        Returns:
            Key ID string
        """
        # Generate random key ID
        random_bytes = secrets.token_bytes(16)
        timestamp = int(utc_now().timestamp())
        
        # Combine timestamp and random bytes
        key_id = f"{timestamp}_{base64.b64encode(random_bytes).decode('utf-8')}"
        
        return key_id
    
    async def _get_data_key(self, key_id: str) -> bytes:
        """
        Retrieve or regenerate data key for given key ID
        
        Args:
            key_id: Key identifier
            
        Returns:
            32-byte data key
        """
        # Check cache first
        if key_id in self._key_cache:
            key, cached_time = self._key_cache[key_id]
            
            # Check if cache entry is still valid
            if utc_now() - cached_time < self._cache_ttl:
                return key
        
        # Regenerate key deterministically from key_id and master key
        # This ensures we can always decrypt old data
        try:
            _, salt_b64 = key_id.split("_", 1)
            salt = base64.b64decode(salt_b64)
        except Exception as e:
            raise EncryptionException(f"Invalid key_id format: {str(e)}")
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        data_key = kdf.derive(self.master_key)
        
        # Update cache
        self._key_cache[key_id] = (data_key, utc_now())
        
        # Clean old cache entries
        self._clean_cache()
        
        return data_key
    
    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = utc_now()
        expired_keys = [
            key_id for key_id, (_, cached_time) in self._key_cache.items()
            if current_time - cached_time > self._cache_ttl
        ]
        
        for key_id in expired_keys:
            del self._key_cache[key_id]
    
    @staticmethod
    def generate_new_master_key() -> str:
        """
        Generate a new master key for initial setup
        
        Returns:
            Base64 encoded 256-bit key
        """
        key = secrets.token_bytes(32)
        return base64.b64encode(key).decode('utf-8')
    
    async def encrypt_field(
        self,
        field_name: str,
        field_value: str,
        user_id: str
    ) -> Tuple[str, str]:
        """
        Encrypt a specific field with user context
        
        Args:
            field_name: Name of the field being encrypted
            field_value: Value to encrypt
            user_id: User identifier for additional authentication
            
        Returns:
            Tuple of (encrypted_value, key_id)
        """
        # Use field name and user ID as additional authenticated data
        aad = f"{field_name}:{user_id}".encode('utf-8')
        
        return await self.encrypt_data(field_value, aad)
    
    async def decrypt_field(
        self,
        field_name: str,
        encrypted_value: str,
        key_id: str,
        user_id: str
    ) -> str:
        """
        Decrypt a specific field with user context
        
        Args:
            field_name: Name of the field being decrypted
            encrypted_value: Encrypted value
            key_id: Key identifier
            user_id: User identifier for additional authentication
            
        Returns:
            Decrypted value
        """
        # Use field name and user ID as additional authenticated data
        aad = f"{field_name}:{user_id}".encode('utf-8')
        
        return await self.decrypt_data(encrypted_value, key_id, aad)

class HashingService:
    """
    Service for hashing sensitive identifiers
    Used for privacy-preserving analytics
    """
    
    def __init__(self, salt: Optional[str] = None):
        """
        Initialize hashing service
        
        Args:
            salt: Salt for hashing. If not provided, reads from environment
        """
        if salt:
            self.salt = salt.encode('utf-8')
        else:
            salt_env = os.environ.get('EMOTIONAL_MEMORY_HASH_SALT')
            if not salt_env:
                # Generate random salt for development
                self.salt = secrets.token_bytes(32)
                logger.warning("Generated random hash salt. This should not happen in production!")
            else:
                self.salt = salt_env.encode('utf-8')
    
    def hash_user_id(self, user_id: str) -> str:
        """
        Hash user ID for privacy-preserving storage
        
        Args:
            user_id: User identifier to hash
            
        Returns:
            Hashed user ID
        """
        # Use SHA-256 with salt
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(self.salt)
        digest.update(user_id.encode('utf-8'))
        
        hash_bytes = digest.finalize()
        return base64.b64encode(hash_bytes).decode('utf-8')
    
    def hash_session_id(self, session_id: str) -> str:
        """
        Hash session ID for privacy
        
        Args:
            session_id: Session identifier to hash
            
        Returns:
            Hashed session ID
        """
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(self.salt)
        digest.update(session_id.encode('utf-8'))
        
        hash_bytes = digest.finalize()
        return base64.b64encode(hash_bytes).decode('utf-8')
    
    def generate_anonymous_id(self, user_id: str, context: str = "") -> str:
        """
        Generate anonymous ID for analytics
        
        Args:
            user_id: User identifier
            context: Additional context for ID generation
            
        Returns:
            Anonymous identifier
        """
        # Combine user ID with context
        data = f"{user_id}:{context}".encode('utf-8')
        
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(self.salt)
        digest.update(data)
        
        hash_bytes = digest.finalize()
        
        # Take first 16 bytes for shorter ID
        anonymous_id = base64.b64encode(hash_bytes[:16]).decode('utf-8')
        
        return anonymous_id.rstrip('=')
